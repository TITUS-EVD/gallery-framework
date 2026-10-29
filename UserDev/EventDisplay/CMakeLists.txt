add_subdirectory(3DViewer)
add_subdirectory(Reader)

# exclude python files we will treat as executable
install(DIRECTORY python/ TYPE BIN PATTERN evd.py EXCLUDE)
install(
    FILES python/evd.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
    DESTINATION bin
)
