import discord
import aiohttp
import os

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

API_URL = "https://monitoring-suspicious-discussions-on-online-plat-production.up.railway.app/predict"
MOD_CHANNEL_NAME = "moderation-log"

@client.event
async def on_ready():
    print(f"Bot connected as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json={"text": message.content}) as resp:
            if resp.status != 200:
                return
            result = await resp.json()

    warning_level = result.get("warning_level", "none")
    confidence    = result.get("confidence_scores", {})
    hate_pct      = round((confidence.get("hate_speech", 0)) * 100, 1)
    off_pct       = round((confidence.get("offensive",   0)) * 100, 1)

    mod_channel = discord.utils.get(message.guild.text_channels, name=MOD_CHANNEL_NAME)

    if warning_level == "high":
        await message.delete()
        await message.channel.send(
            f"⚠️ {message.author.mention} your message was removed. "
            f"Hate speech detected ({hate_pct}% confidence)."
        )
        if mod_channel:
            embed = discord.Embed(title="🚨 HIGH RISK — Message Removed", color=0xe74c3c)
            embed.add_field(name="User", value=str(message.author), inline=True)
            embed.add_field(name="Channel", value=str(message.channel), inline=True)
            embed.add_field(name="Message", value=message.content[:500], inline=False)
            embed.add_field(name="Hate %", value=f"{hate_pct}%", inline=True)
            embed.add_field(name="Method", value=result.get("detection_method", "ml_model"), inline=True)
            await mod_channel.send(embed=embed)

    elif warning_level == "medium":
        if mod_channel:
            embed = discord.Embed(title="⚠️ MEDIUM RISK — Flagged for Review", color=0xe67e22)
            embed.add_field(name="User", value=str(message.author), inline=True)
            embed.add_field(name="Channel", value=str(message.channel), inline=True)
            embed.add_field(name="Message", value=message.content[:500], inline=False)
            embed.add_field(name="Offensive %", value=f"{off_pct}%", inline=True)
            embed.add_field(name="Hate %", value=f"{hate_pct}%", inline=True)
            await mod_channel.send(embed=embed)

    elif warning_level == "low":
        if mod_channel:
            await mod_channel.send(f"📝 Low risk message from {message.author} logged. Hate: {hate_pct}%")

client.run(os.environ.get("DISCORD_TOKEN"))