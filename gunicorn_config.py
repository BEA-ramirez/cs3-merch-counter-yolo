import os

# Render provides PORT through environment variable
bind = "0.0.0.0:{}".format(os.environ.get("PORT", "10000"))

# Keep other settings
workers = 1
timeout = 120
loglevel = 'info'