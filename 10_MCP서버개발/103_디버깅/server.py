from fastmcp import FastMCP

mcp = FastMCP(name="HelloWorld")

@mcp.tool()
def hello_world(name: str):
    """Greets the provided name."""
    return f"Hello {name}"

@mcp.prompt()
def generate_welcome(name: str):
    """Generates a welcome message for the provided name."""
    return f"Welcome, {name}! How can I assist you today?"

@mcp.resource("resource://user-info")
def get_user_info():
    """provides sample user information."""
    return {"username":"guest", "level": "basic"}
   
if __name__ == "__main__":
    mcp.run()
