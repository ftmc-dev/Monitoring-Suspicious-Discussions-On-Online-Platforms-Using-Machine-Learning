# ======================================================================
# DISCOURSEGUARD — Discord Moderation Bot (FIXED)
# ======================================================================

import discord
import aiohttp
import os
from datetime import timedelta, datetime

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

API_URL = "https://monitoring-suspicious-discussions-on-online-plat-production.up.railway.app"
MOD_CHANNEL_NAME = "moderation-log"
PREFIX = "!"

# ── API Helper Functions ─────────────────────────────────────────────

async def api_sync_strike(user_id, username, message, warning_level, action_taken="warning", 
                          hate_score=0, offensive_score=0):
    """Sync a strike to the API database"""
    try:
        async with aiohttp.ClientSession() as session:
            # Add the strike
            await session.post(
                f"{API_URL}/api/strikes",
                json={
                    "user_id": str(user_id),
                    "username": username,
                    "message": message,
                    "warning_level": warning_level,
                    "action_taken": action_taken,
                    "hate_score": hate_score,
                    "offensive_score": offensive_score
                },
                timeout=aiohttp.ClientTimeout(total=5)
            )
            return True
    except Exception as e:
        print(f"API sync error: {e}")
        return False

async def api_get_user_strikes(user_id):
    """Get user strikes from API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{API_URL}/api/users/{user_id}/strikes",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"strikes": 0, "high_strikes": 0, "medium_strikes": 0, "warnings": 0, "status": "active"}
    except Exception:
        return {"strikes": 0, "high_strikes": 0, "medium_strikes": 0, "warnings": 0, "status": "active"}

async def api_update_user_status(user_id, status):
    """Update user status in API"""
    try:
        async with aiohttp.ClientSession() as session:
            await session.put(
                f"{API_URL}/api/users/{user_id}/status",
                json={"status": status},
                timeout=aiohttp.ClientTimeout(total=5)
            )
            return True
    except Exception:
        return False

async def send_dm(user, message):
    """Send a DM with better error handling"""
    try:
        await user.send(message)
        return True
    except discord.Forbidden:
        print(f"Cannot DM {user.name}: DMs disabled")
        return False
    except discord.HTTPException as e:
        print(f"DM failed for {user.name}: {e}")
        return False

# ── Events ────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    print(f"✓ DiscourseGuard connected as {client.user}")
    print(f"✓ Monitoring {len(client.guilds)} server(s)")
    print(f"✓ API: {API_URL}")
    print(f"\nAdmin commands (use in #{MOD_CHANNEL_NAME}):")
    print("  !warn     @user [reason]")
    print("  !timeout  @user [minutes] [reason]")
    print("  !kick     @user [reason]")
    print("  !ban      @user [reason]")
    print("  !unban    user_id")
    print("  !strikes  @user")
    print("  !history  @user")
    print("  !status   @user")

@client.event
async def on_message(message):
    # Ignore bot messages and empty messages
    if message.author.bot:
        return
    if not message.content.strip():
        return

    # Admin commands only in moderation-log
    if message.channel.name == MOD_CHANNEL_NAME:
        if message.content.startswith(PREFIX):
            await handle_admin_command(message)
        return

    # Check if it's a DM (private message)
    if isinstance(message.channel, discord.DMChannel):
        print(f"[DM] {message.author}: {message.content}")
        # Don't analyze DMs, just log them
        return

    # Analyze all other messages
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL}/predict",
                json={
                    "text": message.content,
                    "user_id": str(message.author.id),
                    "username": str(message.author)
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return
                result = await resp.json()
    except Exception as e:
        print(f"API error: {e}")
        return

    warning_level = result.get("warning_level", "none")
    confidence = result.get("confidence_scores", {})
    hate_pct = round(confidence.get("hate_speech", 0) * 100, 1)
    off_pct = round(confidence.get("offensive", 0) * 100, 1)
    normal_pct = round(confidence.get("normal", 0) * 100, 1)
    method = result.get("detection_method", "ml_model")
    is_suspicious = result.get("is_suspicious", False)

    mod_channel = discord.utils.get(
        message.guild.text_channels,
        name=MOD_CHANNEL_NAME
    )

    # ── HIGH RISK ─────────────────────────────────────────────────────
    if warning_level == "high":
        # Get user's strike count from API
        user_data = await api_get_user_strikes(str(message.author.id))
        high_strikes = user_data.get("high_strikes", 0)
        
        # 🔴 DELETE the message so NO ONE sees it
        try:
            await message.delete()
            print(f"🗑️ Deleted message from {message.author}")
        except discord.Forbidden:
            print("❌ Cannot delete message - missing permissions")
        except Exception as e:
            print(f"❌ Delete error: {e}")

        # Determine action
        action_taken = ""
        dm_sent = False
        
        if high_strikes >= 3:
            # Third high strike — auto ban
            try:
                dm_sent = await send_dm(message.author,
                    f"🔨 **PERMANENT BAN**\n"
                    f"You have been **permanently banned** from **{message.guild.name}**.\n"
                    f"**Reason:** Repeated hate speech ({high_strikes + 1} strikes)."
                )
                await message.guild.ban(
                    message.author,
                    reason=f"Auto-ban: {high_strikes + 1} hate speech strikes",
                    delete_message_days=0
                )
                action_taken = f"🔨 **Auto-banned** — {high_strikes + 1} high strikes"
                await api_update_user_status(str(message.author.id), "banned")
            except discord.Forbidden:
                action_taken = "❌ Auto-ban failed — missing permissions"

        elif high_strikes >= 2:
            # Second high strike — auto timeout 24 hours
            try:
                await message.author.timeout(
                    timedelta(hours=24),
                    reason="Second hate speech violation"
                )
                dm_sent = await send_dm(message.author,
                    f"⏱️ **TIMEOUT**\n"
                    f"You have been **timed out for 24 hours** in **{message.guild.name}**.\n"
                    f"**Reason:** Second hate speech violation.\n"
                    f"*A third violation will result in a permanent ban.*"
                )
                action_taken = "⏱️ **Auto-timeout 24h** — second high strike"
                await api_update_user_status(str(message.author.id), "suspended")
            except discord.Forbidden:
                action_taken = "❌ Auto-timeout failed — missing permissions"

        else:
            # First high strike — warn only
            dm_sent = await send_dm(message.author,
                f"⚠️ **WARNING**\n"
                f"Your message in **{message.guild.name}** was removed.\n"
                f"**Reason:** Hate speech detected ({hate_pct}% confidence).\n"
                f"**Message deleted:** \"{message.content[:100]}...\"\n"
                f"*Further violations will result in a timeout or permanent ban.*"
            )
            action_taken = "⚠️ **Warning sent** — first high strike"

        # 🔴 Send notification in the channel (but NOT the original message)
        try:
            await message.channel.send(
                f"🚫 **Message Deleted**\n"
                f"{message.author.mention} - Your message was removed for **hate speech**.\n"
                f"Strike recorded. Check your DMs for details."
            )
        except discord.Forbidden:
            pass

        # Full alert in moderation-log
        if mod_channel:
            embed = discord.Embed(
                title="🚨 HIGH RISK — Message Deleted",
                color=0xe74c3c,
                timestamp=message.created_at
            )
            embed.add_field(
                name="👤 User",
                value=f"{message.author.mention}\nID: `{message.author.id}`",
                inline=True
            )
            embed.add_field(name="📍 Channel", value=message.channel.mention, inline=True)
            embed.add_field(name="⚡ High Strikes", value=f"**{high_strikes + 1}**", inline=True)
            embed.add_field(name="💬 Message (Deleted)", value=f"```{message.content[:500]}```", inline=False)
            embed.add_field(name="🔴 Hate %", value=f"{hate_pct}%", inline=True)
            embed.add_field(name="🟠 Offensive %", value=f"{off_pct}%", inline=True)
            embed.add_field(name="🟢 Normal %", value=f"{normal_pct}%", inline=True)
            embed.add_field(name="⚙️ Method", value=method, inline=True)
            embed.add_field(name="✅ Action Taken", value=action_taken, inline=False)
            embed.add_field(name="📨 DM Sent", value="✅ Yes" if dm_sent else "❌ No (DMs disabled)", inline=True)
            embed.set_footer(text="DiscourseGuard — Automated Escalation System")
            await mod_channel.send(embed=embed)

        print(f"[HIGH] Strike {high_strikes + 1} — {message.author}: {message.content[:50]}")

    # ── MEDIUM RISK ───────────────────────────────────────────────────
    elif warning_level == "medium":
        # ⚠️ Send DM warning
        dm_sent = await send_dm(message.author,
            f"⚠️ **WARNING**\n"
            f"A message you sent in **{message.guild.name}** was flagged.\n"
            f"**Reason:** Offensive content detected ({off_pct}% confidence).\n"
            f"**Message:** \"{message.content[:100]}...\"\n"
            f"*Please keep discussions respectful. Repeated violations will result in action.*"
        )

        # Send notification in channel (not deleting the message)
        try:
            await message.channel.send(
                f"⚠️ {message.author.mention} - This message has been flagged as **offensive**. "
                f"Please keep discussions respectful."
            )
        except discord.Forbidden:
            pass

        if mod_channel:
            embed = discord.Embed(
                title="⚠️ MEDIUM RISK — Warning Sent",
                color=0xe67e22,
                timestamp=message.created_at
            )
            embed.add_field(
                name="👤 User",
                value=f"{message.author.mention}\nID: `{message.author.id}`",
                inline=True
            )
            embed.add_field(name="📍 Channel", value=message.channel.mention, inline=True)
            embed.add_field(name="💬 Message", value=f"```{message.content[:500]}```", inline=False)
            embed.add_field(name="🔴 Hate %", value=f"{hate_pct}%", inline=True)
            embed.add_field(name="🟠 Offensive %", value=f"{off_pct}%", inline=True)
            embed.add_field(name="🟢 Normal %", value=f"{normal_pct}%", inline=True)
            embed.add_field(name="✅ Action", value="⚠️ Auto-warning sent via DM", inline=False)
            embed.add_field(name="📨 DM Sent", value="✅ Yes" if dm_sent else "❌ No (DMs disabled)", inline=True)
            embed.set_footer(text="DiscourseGuard — Admin can escalate manually")
            await mod_channel.send(embed=embed)

        print(f"[MEDIUM] Auto-warned {message.author}: {message.content[:50]}")

    # ── LOW RISK ──────────────────────────────────────────────────────
    elif warning_level == "low":
        if mod_channel:
            await mod_channel.send(
                f"📝 **Low risk** from {message.author.mention} — "
                f"Hate: {hate_pct}% | Offensive: {off_pct}%\n"
                f"Message: *{message.content[:100]}*"
            )
        print(f"[LOW] Logged {message.author}: {message.content[:50]}")

    # ── NORMAL ────────────────────────────────────────────────────────
    else:
        print(f"[OK] {message.author}: {message.content[:50]}")

# ── Admin Command Handler ─────────────────────────────────────────────

async def handle_admin_command(message):
    parts = message.content.split()
    command = parts[0].lower()
    channel = message.channel

    async def get_target():
        if message.mentions:
            return message.mentions[0]
        await channel.send("❌ Please mention a user. Example: `!warn @username reason`")
        return None

    # ── !warn @user [reason] ─────────────────────────────────────────
    if command == "!warn":
        target = await get_target()
        if not target:
            return
        reason = " ".join(parts[2:]) if len(parts) > 2 else "Violation of server rules"

        dm_sent = await send_dm(target,
            f"⚠️ **WARNING**\n"
            f"You have received a warning in **{message.guild.name}**.\n"
            f"**Reason:** {reason}\n"
            f"*Continued violations may result in a timeout or ban.*"
        )
        
        await channel.send(
            f"✅ Warning sent to {target.mention}.\n"
            f"**Reason:** {reason}\n"
            f"📨 DM Sent: {'✅ Yes' if dm_sent else '❌ No (DMs disabled)'}"
        )
        
        # Sync warning to API
        await api_sync_strike(
            str(target.id),
            str(target),
            f"Manual warning: {reason}",
            "medium",
            "warning"
        )

    # ── !timeout @user [minutes] [reason] ────────────────────────────
    elif command == "!timeout":
        target = await get_target()
        if not target:
            return

        minutes = 60
        reason = "Violation of server rules"
        if len(parts) > 2:
            try:
                minutes = int(parts[2])
                reason = " ".join(parts[3:]) if len(parts) > 3 else reason
            except ValueError:
                reason = " ".join(parts[2:])

        try:
            await target.timeout(timedelta(minutes=minutes), reason=reason)
            dm_sent = await send_dm(target,
                f"⏱️ **TIMEOUT**\n"
                f"You have been **timed out for {minutes} minutes** "
                f"in **{message.guild.name}**.\n**Reason:** {reason}"
            )
            await channel.send(
                f"⏱️ {target.mention} timed out for **{minutes} minutes**.\n"
                f"**Reason:** {reason}\n"
                f"📨 DM Sent: {'✅ Yes' if dm_sent else '❌ No (DMs disabled)'}"
            )
            await api_update_user_status(str(target.id), "suspended")
        except discord.Forbidden:
            await channel.send("❌ Missing permission to timeout this user.")
        except Exception as e:
            await channel.send(f"❌ Error: {e}")

    # ── !kick @user [reason] ─────────────────────────────────────────
    elif command == "!kick":
        target = await get_target()
        if not target:
            return
        reason = " ".join(parts[2:]) if len(parts) > 2 else "Violation of server rules"

        dm_sent = await send_dm(target,
            f"👢 **KICKED**\n"
            f"You have been **kicked** from **{message.guild.name}**.\n"
            f"**Reason:** {reason}"
        )
        try:
            await message.guild.kick(target, reason=reason)
            await channel.send(
                f"👢 {target.mention} has been kicked.\n"
                f"**Reason:** {reason}\n"
                f"📨 DM Sent: {'✅ Yes' if dm_sent else '❌ No (DMs disabled)'}"
            )
            await api_update_user_status(str(target.id), "active")
        except discord.Forbidden:
            await channel.send("❌ Missing permission to kick this user.")

    # ── !ban @user [reason] ──────────────────────────────────────────
    elif command == "!ban":
        target = await get_target()
        if not target:
            return
        reason = " ".join(parts[2:]) if len(parts) > 2 else "Violation of server rules"

        dm_sent = await send_dm(target,
            f"🔨 **BANNED**\n"
            f"You have been **permanently banned** from **{message.guild.name}**.\n"
            f"**Reason:** {reason}"
        )
        try:
            await message.guild.ban(
                target, reason=reason, delete_message_days=0
            )
            await channel.send(
                f"🔨 {target.mention} has been banned.\n"
                f"**Reason:** {reason}\n"
                f"📨 DM Sent: {'✅ Yes' if dm_sent else '❌ No (DMs disabled)'}"
            )
            await api_update_user_status(str(target.id), "banned")
        except discord.Forbidden:
            await channel.send("❌ Missing permission to ban this user.")

    # ── !unban user_id ────────────────────────────────────────────────
    elif command == "!unban":
        if len(parts) < 2:
            await channel.send(
                "❌ Usage: `!unban 123456789` (use the user's ID number)\n"
                "To find the ID: Developer Mode → right-click user → Copy ID"
            )
            return
        try:
            user_id = int(parts[1].strip("<@!>"))
            await message.guild.unban(discord.Object(id=user_id))
            await channel.send(f"✅ User `{user_id}` has been unbanned.")
            await api_update_user_status(str(user_id), "active")
        except ValueError:
            await channel.send("❌ Invalid ID. Use numbers only. Example: `!unban 123456789`")
        except discord.NotFound:
            await channel.send("❌ That user is not banned.")
        except discord.Forbidden:
            await channel.send("❌ Missing permission to unban users.")

    # ── !strikes @user ───────────────────────────────────────────────
    elif command == "!strikes":
        target = await get_target()
        if not target:
            return

        user_data = await api_get_user_strikes(str(target.id))
        
        if user_data["strikes"] == 0:
            await channel.send(f"✅ {target.mention} has no recorded strikes.")
            return

        embed = discord.Embed(
            title=f"📊 Strike History — {target.name}",
            color=0x3498db
        )
        embed.add_field(name="Total Strikes", value=user_data["strikes"], inline=True)
        embed.add_field(name="High Strikes", value=user_data["high_strikes"], inline=True)
        embed.add_field(name="Medium Strikes", value=user_data["medium_strikes"], inline=True)
        embed.add_field(name="Warnings Issued", value=user_data["warnings"], inline=True)
        embed.add_field(name="Status", value=user_data["status"].upper(), inline=True)

        # Escalation status
        high = user_data["high_strikes"]
        if high >= 3:
            embed.add_field(name="🚨 Status", value="Should be banned", inline=False)
        elif high == 2:
            embed.add_field(name="⚠️ Status", value="Should be on timeout", inline=False)
        elif high == 1:
            embed.add_field(name="⚠️ Status", value="Warned — one more high strike = timeout", inline=False)
        else:
            embed.add_field(name="✅ Status", value="No high strikes yet", inline=False)

        await channel.send(embed=embed)

    # ── !history @user ───────────────────────────────────────────────
    elif command == "!history":
        target = await get_target()
        if not target:
            return

        # Get strikes from API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{API_URL}/api/strikes?limit=50",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        user_strikes = [s for s in data["strikes"] if s["user_id"] == str(target.id)]
                        
                        if not user_strikes:
                            await channel.send(f"✅ {target.mention} has no flagged message history.")
                            return
                        
                        embed = discord.Embed(
                            title=f"📋 Message History — {target.name}",
                            color=0x9b59b6
                        )
                        for i, s in enumerate(user_strikes[:10], 1):
                            embed.add_field(
                                name=f"#{i} [{s['warning_level'].upper()}]",
                                value=f"{s['message'][:100]}...",
                                inline=False
                            )
                        await channel.send(embed=embed)
        except Exception as e:
            await channel.send(f"❌ Error fetching history: {e}")

    # ── !status @user ─────────────────────────────────────────────────
    elif command == "!status":
        target = await get_target()
        if not target:
            return
        
        user_data = await api_get_user_strikes(str(target.id))
        embed = discord.Embed(
            title=f"👤 User Status — {target.name}",
            color=0x00b894
        )
        embed.add_field(name="User ID", value=f"`{target.id}`", inline=True)
        embed.add_field(name="Status", value=user_data["status"].upper(), inline=True)
        embed.add_field(name="Total Strikes", value=user_data["strikes"], inline=True)
        embed.add_field(name="High Strikes", value=user_data["high_strikes"], inline=True)
        embed.add_field(name="Medium Strikes", value=user_data["medium_strikes"], inline=True)
        embed.add_field(name="Warnings", value=user_data["warnings"], inline=True)
        await channel.send(embed=embed)

    # ── !help ─────────────────────────────────────────────────────────
    elif command == "!help":
        embed = discord.Embed(
            title="🛡️ DiscourseGuard — Admin Commands",
            color=0x6c5ce7
        )
        embed.add_field(
            name="⚠️ !warn @user [reason]",
            value="Send a warning DM to the user",
            inline=False
        )
        embed.add_field(
            name="⏱️ !timeout @user [minutes] [reason]",
            value="Mute user for X minutes (default 60)",
            inline=False
        )
        embed.add_field(
            name="👢 !kick @user [reason]",
            value="Remove user from server (can rejoin)",
            inline=False
        )
        embed.add_field(
            name="🔨 !ban @user [reason]",
            value="Permanently ban user from server",
            inline=False
        )
        embed.add_field(
            name="✅ !unban user_id",
            value="Unban a user using their Discord ID number",
            inline=False
        )
        embed.add_field(
            name="📊 !strikes @user",
            value="Show strike count and escalation status",
            inline=False
        )
        embed.add_field(
            name="📋 !history @user",
            value="Show last 10 flagged messages from user",
            inline=False
        )
        embed.add_field(
            name="👤 !status @user",
            value="Show user's current status and full stats",
            inline=False
        )
        embed.add_field(
            name="🤖 Automatic escalation",
            value="High strike 1 → Warning\nHigh strike 2 → Timeout 24h\nHigh strike 3+ → Auto-ban",
            inline=False
        )
        embed.set_footer(text="DiscourseGuard — Hate Speech Detection System")
        await channel.send(embed=embed)

    else:
        await channel.send(
            f"❓ Unknown command. Type `!help` to see all available commands."
        )

# ── Run ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

token = os.environ.get("DISCORD_TOKEN")
if not token:
    raise ValueError("DISCORD_TOKEN environment variable not set")

client.run(token)