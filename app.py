# ======================================================================
# HATE SPEECH DETECTION API - Flask with Database
# ======================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import re
import joblib
from datetime import datetime
import numpy as np

model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
print(model.classes_)

app = Flask(__name__)
CORS(app)  # Allow all origins

# ── Database Setup ─────────────────────────────────────────────────────

def init_db():
    """Initialize SQLite database with all tables"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        
        # Strikes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS strikes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                warning_level TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action_taken TEXT DEFAULT 'warning',
                reviewed INTEGER DEFAULT 0,
                hate_score REAL DEFAULT 0,
                offensive_score REAL DEFAULT 0,
                detection_method TEXT DEFAULT 'ml_model',
                matched_keyword TEXT
            )
        ''')
        
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                strikes INTEGER DEFAULT 0,
                warnings INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                last_seen TEXT,
                high_strikes INTEGER DEFAULT 0,
                medium_strikes INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

# Initialize database on startup
init_db()

# ── Routes ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "DiscourseGuard API - Hate Speech Detection",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "GET /api/strikes": "Get all strikes",
            "POST /api/strikes": "Add a strike",
            "PUT /api/strikes/<id>/review": "Mark strike as reviewed",
            "DELETE /api/strikes/<id>": "Delete a strike",
            "GET /api/users": "Get all users",
            "PUT /api/users/<user_id>/status": "Update user status",
            "GET /api/users/<user_id>/strikes": "Get user strikes",
            "GET /api/stats": "Get statistics",
            "GET /api/db-status": "Check database status",
            "POST /predict": "Analyze text for hate speech"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "database": "SQLite",
        "database_file": "moderation.db",
        "timestamp": datetime.now().isoformat()
    })

# ── STRIKES ENDPOINTS ────────────────────────────────────────────────

@app.route("/api/strikes", methods=["GET"])
def api_get_strikes():
    """Get all strikes"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        c.execute('''
            SELECT id, user_id, username, message, warning_level, timestamp, 
                   action_taken, reviewed, hate_score, offensive_score, 
                   detection_method, matched_keyword
            FROM strikes 
            ORDER BY timestamp DESC 
            LIMIT 200
        ''')
        
        strikes = []
        for row in c.fetchall():
            strikes.append({
                "id": row[0],
                "user_id": row[1],
                "username": row[2],
                "message": row[3],
                "warning_level": row[4],
                "timestamp": row[5],
                "action_taken": row[6],
                "reviewed": bool(row[7]),
                "hate_score": row[8] if row[8] is not None else 0,
                "offensive_score": row[9] if row[9] is not None else 0,
                "detection_method": row[10] if row[10] is not None else "ml_model",
                "matched_keyword": row[11]
            })
        conn.close()
        
        return jsonify({
            "total": len(strikes),
            "strikes": strikes
        })
    except Exception as e:
        print(f"❌ Error getting strikes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/strikes", methods=["POST"])
def api_add_strike():
    """Add a strike"""
    try:
        data = request.json
        print(f"📥 Received strike data: {data}")
        
        required = ["user_id", "username", "message", "warning_level"]
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields"}), 400
        
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        
        # Insert strike
        c.execute('''
            INSERT INTO strikes 
            (user_id, username, message, warning_level, timestamp, action_taken, 
             hate_score, offensive_score, detection_method, matched_keyword)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["user_id"],
            data["username"],
            data["message"],
            data["warning_level"],
            datetime.now().isoformat(),
            data.get("action_taken", "warning"),
            float(data.get("hate_score", 0)),
            float(data.get("offensive_score", 0)),
            data.get("detection_method", "ml_model"),
            data.get("matched_keyword", None)
        ))
        
        # Update or insert user
        c.execute('''
            INSERT INTO users (user_id, username, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                last_seen = excluded.last_seen
        ''', (data["user_id"], data["username"], datetime.now().isoformat()))
        
        # Update strike counts based on warning level
        if data["warning_level"] == "high":
            c.execute('''
                UPDATE users 
                SET strikes = strikes + 1, 
                    high_strikes = high_strikes + 1 
                WHERE user_id = ?
            ''', (data["user_id"],))
        elif data["warning_level"] == "medium":
            c.execute('''
                UPDATE users 
                SET strikes = strikes + 1, 
                    medium_strikes = medium_strikes + 1 
                WHERE user_id = ?
            ''', (data["user_id"],))
        else:
            c.execute('''
                UPDATE users 
                SET strikes = strikes + 1 
                WHERE user_id = ?
            ''', (data["user_id"],))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Strike saved for {data['username']} ({data['warning_level']})")
        return jsonify({"status": "success", "message": "Strike saved"})
        
    except Exception as e:
        print(f"❌ Error saving strike: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/strikes/<int:strike_id>/review", methods=["PUT"])
def api_review_strike(strike_id):
    """Mark a strike as reviewed"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        c.execute('UPDATE strikes SET reviewed = 1 WHERE id = ?', (strike_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Strike {strike_id} marked as reviewed"})
    except Exception as e:
        print(f"❌ Error reviewing strike: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/strikes/<int:strike_id>", methods=["DELETE"])
def api_delete_strike(strike_id):
    """Delete a strike"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        c.execute('DELETE FROM strikes WHERE id = ?', (strike_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Strike {strike_id} deleted"})
    except Exception as e:
        print(f"❌ Error deleting strike: {e}")
        return jsonify({"error": str(e)}), 500

# ── USERS ENDPOINTS ───────────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Get all users"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        c.execute('''
            SELECT user_id, username, strikes, warnings, status, last_seen, 
                   high_strikes, medium_strikes
            FROM users
            ORDER BY strikes DESC
        ''')
        
        users = {}
        for row in c.fetchall():
            users[row[0]] = {
                "username": row[1],
                "strikes": row[2],
                "warnings": row[3],
                "status": row[4] if row[4] else "active",
                "last_seen": row[5],
                "high_strikes": row[6] if row[6] else 0,
                "medium_strikes": row[7] if row[7] else 0
            }
        conn.close()
        
        return jsonify({
            "total": len(users),
            "users": users
        })
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<user_id>/status", methods=["PUT"])
def api_update_user_status(user_id):
    """Update user status"""
    try:
        data = request.json
        if "status" not in data:
            return jsonify({"error": "Missing status field"}), 400
        
        valid_statuses = ["active", "warned", "suspended", "banned"]
        if data["status"] not in valid_statuses:
            return jsonify({"error": f"Invalid status. Must be one of: {valid_statuses}"}), 400
        
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        c.execute('UPDATE users SET status = ? WHERE user_id = ?', 
                 (data["status"], str(user_id)))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"User {user_id} status updated to {data['status']}"})
    except Exception as e:
        print(f"❌ Error updating user status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<user_id>/strikes", methods=["GET"])
def api_get_user_strikes(user_id):
    """Get strikes for a specific user"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        c.execute('''
            SELECT strikes, warnings, status, high_strikes, medium_strikes 
            FROM users WHERE user_id = ?
        ''', (str(user_id),))
        result = c.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                "strikes": result[0] if result[0] else 0,
                "warnings": result[1] if result[1] else 0,
                "status": result[2] if result[2] else "active",
                "high_strikes": result[3] if result[3] else 0,
                "medium_strikes": result[4] if result[4] else 0
            })
        return jsonify({
            "strikes": 0,
            "warnings": 0,
            "status": "active",
            "high_strikes": 0,
            "medium_strikes": 0
        })
    except Exception as e:
        print(f"❌ Error getting user strikes: {e}")
        return jsonify({"error": str(e)}), 500

# ── STATISTICS ENDPOINT ──────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def api_get_stats():
    """Get overall statistics"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        
        # Count total strikes
        c.execute("SELECT COUNT(*) FROM strikes")
        result = c.fetchone()
        total = result[0] if result else 0
        
        # Count high risk
        c.execute("SELECT COUNT(*) FROM strikes WHERE warning_level = 'high'")
        result = c.fetchone()
        high = result[0] if result else 0
        
        # Count medium risk
        c.execute("SELECT COUNT(*) FROM strikes WHERE warning_level = 'medium'")
        result = c.fetchone()
        medium = result[0] if result else 0
        
        # Count low risk
        c.execute("SELECT COUNT(*) FROM strikes WHERE warning_level = 'low'")
        result = c.fetchone()
        low = result[0] if result else 0
        
        # Count reviewed
        c.execute("SELECT COUNT(*) FROM strikes WHERE reviewed = 1")
        result = c.fetchone()
        reviewed = result[0] if result else 0
        
        # Count users
        c.execute("SELECT COUNT(*) FROM users")
        result = c.fetchone()
        users = result[0] if result else 0
        
        conn.close()
        
        return jsonify({
            "total_strikes": total,
            "high": high,
            "medium": medium,
            "low": low,
            "reviewed": reviewed,
            "total_users": users,
            "pending": total - reviewed
        })
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500

# ── DATABASE STATUS ENDPOINT ─────────────────────────────────────────

@app.route("/api/db-status", methods=["GET"])
def api_db_status():
    """Check database status"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), 'moderation.db')
        
        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            
            # Count tables
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in c.fetchall()]
            conn.close()
            
            return jsonify({
                "exists": True,
                "path": db_path,
                "size_bytes": size,
                "size_kb": round(size / 1024, 2),
                "tables": tables,
                "message": "✅ Database is working!"
            })
        else:
            return jsonify({
                "exists": False,
                "path": db_path,
                "message": "Database not created yet. It will be created when data is saved."
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── TEST ENDPOINTS ────────────────────────────────────────────────────

@app.route("/api/test", methods=["GET"])
def api_test():
    """Test endpoint"""
    return jsonify({
        "status": "success",
        "message": "API is working!",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/test-save", methods=["GET"])
def api_test_save():
    """Test saving data to database"""
    try:
        conn = sqlite3.connect('moderation.db')
        c = conn.cursor()
        
        # Insert test data
        c.execute('''
            INSERT INTO strikes (user_id, username, message, warning_level, timestamp, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('test_123', 'TestUser', 'This is a test message from the API', 'high', datetime.now().isoformat(), 'warning'))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Test data saved to database! Check /api/strikes"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ── PREDICTION ENDPOINT ──────────────────────────────────────────────

# ======================================================================
# Add these two imports/loads near the TOP of app.py, with your other
# imports (you already have "import joblib", just add the load lines)
# ======================================================================

import joblib
model = joblib.load('models/best_model.pkl')

# ======================================================================
# Replace your ENTIRE existing /predict route with this whole function
# ======================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """Analyze text for hate speech"""
    try:
        data = request.json

        if not data or "text" not in data:
            return jsonify({"error": "Missing text field"}), 400

        text = data["text"]
        text_lower = text.lower()

        # ── Layer 1: Keyword rule check (fast path for obvious cases) ──
        hate_keywords = ['kill all', 'exterminate', 'subhuman']
        matched_keyword = next((kw for kw in hate_keywords if kw in text_lower), None)

        if matched_keyword:
            label_id = 2
            warning_level = "high"
            prediction = "Hate Speech"
            hate_score = 1.0
            offensive_score = 0.0
            normal_score = 0.0
            detection_method = "rule_layer"

        else:
            # ── Layer 2: ML model (Logistic Regression) ──
            X = vectorizer.transform([text_lower])
            probabilities = model.predict_proba(X)[0]

            # Confirmed mapping: 0=Normal, 1=Offensive, 2=Hate Speech
            normal_score    = float(probabilities[0])
            offensive_score = float(probabilities[1])
            hate_score      = float(probabilities[2])

            predicted_index = int(probabilities.argmax())
            label_map = {
                0: (0, "none",   "Normal"),
                1: (1, "medium", "Offensive"),
                2: (2, "high",   "Hate Speech")
            }
            label_id, warning_level, prediction = label_map[predicted_index]
            detection_method = "ml_model"

        result = {
            "prediction": prediction,
            "label": ["normal", "offensive", "hate_speech"][label_id],
            "label_id": label_id,
            "is_suspicious": label_id in [1, 2],
            "warning_level": warning_level,
            "confidence_scores": {
                "normal": round(normal_score, 4),
                "offensive": round(offensive_score, 4),
                "hate_speech": round(hate_score, 4)
            },
            "original_text": text,
            "preprocessed_text": text_lower,
            "detection_method": detection_method,
            "matched_keyword": matched_keyword
        }

        # ── Save to database if user_id provided (unchanged logic, just
        #    now uses the hate_score/offensive_score defined above) ──
        if data.get("user_id") and data.get("username"):
            user_id = data["user_id"]
            username = data["username"]

            if warning_level in ["high", "medium"]:
                conn = sqlite3.connect('moderation.db')
                c = conn.cursor()

                c.execute('''
                    INSERT INTO strikes
                    (user_id, username, message, warning_level, timestamp, action_taken,
                     hate_score, offensive_score, detection_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    username,
                    text,
                    warning_level,
                    datetime.now().isoformat(),
                    "warning",
                    hate_score,
                    offensive_score,
                    detection_method
                ))

                c.execute('''
                    INSERT INTO users (user_id, username, last_seen)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username = excluded.username,
                        last_seen = excluded.last_seen
                ''', (user_id, username, datetime.now().isoformat()))

                if warning_level == "high":
                    c.execute('UPDATE users SET strikes = strikes + 1, high_strikes = high_strikes + 1 WHERE user_id = ?', (user_id,))
                else:
                    c.execute('UPDATE users SET strikes = strikes + 1, medium_strikes = medium_strikes + 1 WHERE user_id = ?', (user_id,))

                conn.commit()
                conn.close()
                print(f"✅ Saved prediction for {username} ({warning_level})")

        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("   🛡️  DISCOURSEGUARD API")
    print("   Hate Speech Detection & Moderation System")
    print("=" * 65)
    
    # Check database
    db_path = os.path.abspath('moderation.db')
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"   ✅ Database: {db_path}")
        print(f"   📊 Size: {size} bytes ({round(size/1024, 2)} KB)")
    else:
        print(f"   ⚠️  Database not found. Will be created on first save.")
    
    print("=" * 65)
    print("   🚀 API running on http://127.0.0.1:5000")
    print("   📋 Endpoints:")
    print("      GET  /              - API info")
    print("      GET  /health        - Health check")
    print("      GET  /api/strikes   - Get all strikes")
    print("      POST /api/strikes   - Add a strike")
    print("      GET  /api/users     - Get all users")
    print("      GET  /api/stats     - Get statistics")
    print("      GET  /api/db-status - Database status")
    print("      POST /predict       - Analyze text")
    print("=" * 65)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)