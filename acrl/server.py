from tmrl.networking import Server
import tmrl.config.config_constants as cfg

# (NB: When you omit arguments,
# tmrl retrieves the default in your config.json file.
# Read the documentation of each class for more info.)

# === TMRL Server ======================================================================================================

# The TMRL Server is the central point of communication between TMRL entities.
# The Trainer and the RolloutWorkers connect to the Server.

security = cfg.SECURITY  # OK for secure local networks. On the Internet, prefer "TLS".
password = cfg.PASSWORD  # Password defined in TmrlData/config/config.json

server_ip = cfg.PUBLIC_IP_SERVER  # Use your public IP to run over the Internet.
server_port = cfg.PORT  # Ensure this port is reachable when hosting over the Internet.

if __name__ == "__main__":
    # Instantiating a TMRL Server is straightforward.
    # More arguments are available for TLS etc.; see the TMRL documentation.
    my_server = Server(security=security, password=password, port=server_port)