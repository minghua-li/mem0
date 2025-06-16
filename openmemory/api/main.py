import json
import os
from fastapi import FastAPI
from app.database import engine, Base, SessionLocal
from app.mcp_server import setup_mcp_server
from app.routers import memories_router, apps_router, stats_router, config_router
from fastapi_pagination import add_pagination
from fastapi.middleware.cors import CORSMiddleware
from app.models import User, App, Config as ConfigModel
from uuid import uuid4
from app.config import USER_ID, DEFAULT_APP_ID
import datetime

app = FastAPI(title="OpenMemory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create all tables
Base.metadata.create_all(bind=engine)

# Check for USER_ID and create default user if needed
def create_default_user():
    db = SessionLocal()
    try:
        # Check if user exists
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            # Create default user
            user = User(
                id=uuid4(),
                user_id=USER_ID,
                name="Default User",
                created_at=datetime.datetime.now(datetime.UTC)
            )
            db.add(user)
            db.commit()
    finally:
        db.close()


def create_default_app():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            return

        # Check if app already exists
        existing_app = db.query(App).filter(
            App.name == DEFAULT_APP_ID,
            App.owner_id == user.id
        ).first()

        if existing_app:
            return

        app = App(
            id=uuid4(),
            name=DEFAULT_APP_ID,
            owner_id=user.id,
            created_at=datetime.datetime.now(datetime.UTC),
            updated_at=datetime.datetime.now(datetime.UTC),
        )
        db.add(app)
        db.commit()
    finally:
        db.close()

def load_config_from_file():
    """Load configuration from config.json file into database on startup."""
    config_file_path = "config.json"
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            db = SessionLocal()
            try:
                # Check if config already exists
                existing_config = db.query(ConfigModel).filter(ConfigModel.key == "main").first()
                
                if not existing_config:
                    # Create new config entry
                    db_config = ConfigModel(key="main", value=config_data)
                    db.add(db_config)
                    db.commit()
                    print(f"Loaded configuration from {config_file_path} into database")
                else:
                    print("Configuration already exists in database")
            finally:
                db.close()
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file_path}: {e}")
    else:
        print(f"Config file {config_file_path} not found")

# Create default user on startup
create_default_user()
create_default_app()

# Load config from file on startup
load_config_from_file()

# Setup MCP server
setup_mcp_server(app)

# Include routers
app.include_router(memories_router)
app.include_router(apps_router)
app.include_router(stats_router)
app.include_router(config_router)

# Add pagination support
add_pagination(app)
