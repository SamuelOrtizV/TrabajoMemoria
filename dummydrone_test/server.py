from tmrl.networking import Server
import tmrl.config.config_constants as cfg

# (NB: When you omit arguments,
# tmrl retrieves the default in your config.json file.
# Read the documentation of each class for more info.)

# === TMRL Server ======================================================================================================

# The TMRL Server is the central point of communication between TMRL entities.
# The Trainer and the RolloutWorkers connect to the Server.

security = None  # This is fine for secure local networks. On the Internet, use "TLS" instead.
password = cfg.PASSWORD  # This is the password defined in TmrlData/config/config.json

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if you want to run on the Internet.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.

if __name__ == "__main__":
    # Instantiating a TMRL Server is straightforward.
    # More arguments are available for, e.g., using TLS. Please refer to the TMRL documentation.
    my_server = Server(security=security, password=password, port=server_port)