import os
from datetime import timedelta


class Config:
    """Base config — shared across all environments."""

    # ── App ──────────────────────────────────────────────────────
    APP_NAME   = "Image Classifier"
    SECRET_KEY = os.getenv("SECRET_KEY")

    # ── Model ────────────────────────────────────────────────────
    MODEL_NAME     = "MobileNetV2"
    IMG_SIZE       = (224, 224)          # MobileNetV2 requirement
    TOP_K          = 3                   # number of predictions to return

    # ── Upload ───────────────────────────────────────────────────
    MAX_FILE_BYTES     = 5 * 1024 * 1024     # 5 MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_MIMES      = {"image/jpeg", "image/png", "image/webp"}

    # ── Logging ──────────────────────────────────────────────────
    LOG_LEVEL = "INFO"
    LOG_FILE  = "app.log"

    # ── Security ─────────────────────────────────────────────────
    SESSION_COOKIE_HTTPONLY    = True
    SESSION_COOKIE_SAMESITE    = "Lax"
    SESSION_COOKIE_SECURE      = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

    @classmethod
    def validate(cls):
        """
        Called once at startup. Raises EnvironmentError with all
        problems listed so you fix everything in one restart.
        """
        errors = []

        if not cls.SECRET_KEY:
            errors.append("SECRET_KEY env var is not set.")

        if cls.IMG_SIZE[0] != cls.IMG_SIZE[1]:
            errors.append(
                f"IMG_SIZE must be square for MobileNetV2, got {cls.IMG_SIZE}."
            )

        if cls.TOP_K < 1:
            errors.append(f"TOP_K must be at least 1, got {cls.TOP_K}.")

        if cls.MAX_FILE_BYTES < 1024:
            errors.append(f"MAX_FILE_BYTES is suspiciously small: {cls.MAX_FILE_BYTES}.")

        if errors:
            raise EnvironmentError(
                "Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)
            )


class DevelopmentConfig(Config):
    DEBUG      = True
    TESTING    = False
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-key")
    LOG_LEVEL  = "DEBUG"
    TOP_K      = 3


class ProductionConfig(Config):
    DEBUG     = False
    TESTING   = False
    LOG_LEVEL = "WARNING"

    # SECRET_KEY must come from environment — no fallback
    SESSION_COOKIE_SECURE  = True
    SESSION_COOKIE_SAMESITE = "Strict"


class TestingConfig(Config):
    TESTING    = True
    DEBUG      = True
    SECRET_KEY = "test-secret-key"
    LOG_LEVEL  = "WARNING"
    LOG_FILE   = "test.log"
    TOP_K      = 1                      # keep test output minimal


# ── Registry ─────────────────────────────────────────────────────
config = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
    "default":     DevelopmentConfig,
}


def get_config() -> type[Config]:
    """Resolve active config class from FLASK_ENV (defaults to development)."""
    env = os.getenv("FLASK_ENV", "development").lower()
    return config.get(env, DevelopmentConfig)