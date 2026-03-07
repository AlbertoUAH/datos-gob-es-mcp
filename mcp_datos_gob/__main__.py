from mcp_datos_gob.server import mcp


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
