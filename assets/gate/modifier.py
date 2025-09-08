from pxr import Usd, Sdf

def modify_texture_paths(usd_path, new_texture_path, output_path=None, extensions=None):
    """
    Modifies all texture file paths in a USD file.

    Args:
        usd_path (str): Path to the input USD file.
        new_texture_path (str): The new texture path to set.
        output_path (str): Path to save the modified USD. If None, modifies in place.
        extensions (list): List of texture file extensions to search for.
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".exr"]

    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_path}")

    print(f"Opened USD: {usd_path}")
    modified = False

    # Traverse all prims in the stage
    for prim in stage.Traverse():
        # Iterate through all attributes of the prim
        for attr in prim.GetAttributes():
            # Only check attributes that store asset paths
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attr.Get()
                if value and isinstance(value.path, str):
                    # Check if the attribute points to a texture file
                    if any(value.path.lower().endswith(ext) for ext in extensions):
                        print(f"Changing texture on {prim.GetPath()} -> {attr.GetName()}")
                        print(f"   Old path: {value.path}")
                        print(f"   New path: {new_texture_path}")
                        # Set the new texture path
                        attr.Set(Sdf.AssetPath(new_texture_path))
                        modified = True

    # Save the modified USD
    if modified:
        if output_path:
            stage.Export(output_path)
            print(f"Modified USD saved at: {output_path}")
        else:
            stage.GetRootLayer().Save()
            print(f"USD updated in place: {usd_path}")
    else:
        print("No texture references found to modify.")


# Example usage
usd_file = "gate.usd"
new_texture = "/home/uliege/Desktop/drone_track/assets/gate/textures/bitmap.png"  # Can be absolute or relative
modify_texture_paths(usd_file, new_texture)

