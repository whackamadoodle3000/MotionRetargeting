import pickle
import sys
import io # Used for checking object size without loading everything (if needed, but load is simpler here)

# --- Configuration ---
FILENAME = "res.pk"
MAX_DEPTH = 3          # How deep to explore nested structures (e.g., list within dict within list)
MAX_ITEMS = 5          # Max list/dict/set items to show per level
MAX_STR_LEN = 100      # Max length of strings/bytes to display fully before truncating

def print_structure(obj, current_depth=0, max_depth=MAX_DEPTH):
    """
    Recursively prints the structure and a sample of a Python object.
    Limits depth and items per collection to avoid excessive output.
    """
    indent = "  " * current_depth
    obj_type = type(obj)

    # Stop recursion if maximum depth is exceeded
    if current_depth > max_depth:
        print(f"{indent}[... maximum depth ({max_depth}) reached ...]")
        return

    # --- Handle Different Types ---

    # Basic Scalar Types
    if isinstance(obj, (int, float, bool, type(None))):
        print(f"{indent}Type: {obj_type.__name__}, Value: {obj!r}")

    # String or Bytes
    elif isinstance(obj, (str, bytes)):
        length = len(obj)
        type_name = obj_type.__name__
        print(f"{indent}Type: {type_name}, Length: {length}")
        if length > MAX_STR_LEN:
            # Show truncated representation using repr() for clarity
            truncated_repr = repr(obj[:MAX_STR_LEN])
            # Remove the closing quote/apostrophe if repr added one
            if truncated_repr.endswith("'") or truncated_repr.endswith('"'):
                truncated_repr = truncated_repr[:-1]
            # Indicate truncation
            display_val = f"{truncated_repr} ... (truncated)]"
            # Add back the correct prefix for bytes
            if isinstance(obj, bytes):
                 # Ensure 'b' prefix if original repr had it and it got cut off
                if not display_val.startswith("b'") and not display_val.startswith('b"'):
                    display_val = "b'" + display_val if "'" in display_val else 'b"' + display_val
            else:
                 # Ensure quote prefix if original repr had it and it got cut off
                if not display_val.startswith("'") and not display_val.startswith('"'):
                    display_val = "'" + display_val if "'" in display_val else '"' + display_val

            print(f"{indent}  Value: {display_val}")
        else:
            # Use repr() to show quotes and special characters clearly
            print(f"{indent}  Value: {obj!r}")

    # List or Tuple
    elif isinstance(obj, (list, tuple)):
        length = len(obj)
        type_name = obj_type.__name__
        print(f"{indent}Type: {type_name}, Length: {length}")
        if length > 0:
            items_to_show = min(length, MAX_ITEMS)
            print(f"{indent}  First {items_to_show} items:")
            for i, item in enumerate(obj[:items_to_show]):
                print(f"{indent}  [{i}]:")
                print_structure(item, current_depth + 1, max_depth)
            if length > MAX_ITEMS:
                print(f"{indent}  [...] ({length - MAX_ITEMS} more items)")
        else:
             print(f"{indent}  (empty)")


    # Dictionary
    elif isinstance(obj, dict):
        length = len(obj)
        print(f"{indent}Type: dict, Size: {length}")
        if length > 0:
            items_to_show = min(length, MAX_ITEMS)
            print(f"{indent}  First {items_to_show} key-value pairs:")
            count = 0
            for key, value in obj.items():
                if count >= items_to_show:
                    break
                # Display key (truncated if necessary)
                key_repr = repr(key)
                if len(key_repr) > MAX_STR_LEN:
                    key_repr = key_repr[:MAX_STR_LEN] + "...]"
                print(f"{indent}  Key: {key_repr}")
                # Display value structure
                print_structure(value, current_depth + 1, max_depth)
                count += 1
            if length > MAX_ITEMS:
                print(f"{indent}  [...] ({length - MAX_ITEMS} more key-value pairs)")
        else:
             print(f"{indent}  (empty)")

    # Set
    elif isinstance(obj, set):
        length = len(obj)
        print(f"{indent}Type: set, Size: {length}")
        if length > 0:
            items_to_show = min(length, MAX_ITEMS)
            print(f"{indent}  First {items_to_show} items (order not guaranteed):")
            # Convert set to list for slicing, but iterate carefully
            temp_list = list(obj)
            for i in range(items_to_show):
                 item = temp_list[i]
                 print(f"{indent}  Item:")
                 print_structure(item, current_depth + 1, max_depth)

            if length > MAX_ITEMS:
                print(f"{indent}  [...] ({length - MAX_ITEMS} more items)")
        else:
             print(f"{indent}  (empty)")

    # Other object types (like custom classes)
    else:
        print(f"{indent}Type: {obj_type.__name__}")
        try:
            # Try showing a limited representation
            obj_repr = repr(obj)
            if len(obj_repr) > (MAX_STR_LEN + 20): # Allow slightly longer reprs
                obj_repr = obj_repr[:MAX_STR_LEN + 20] + "...]"
            print(f"{indent}  Repr: {obj_repr}")
        except Exception as e:
            print(f"{indent}  (Could not get representation: {e})")
        # You could optionally try printing vars(obj).items() here like the dict
        # but be cautious as it can be very verbose or recurse unexpectedly.
        # if hasattr(obj, '__dict__'):
        #     print(f"{indent}  Attributes (sample):")
        #     # Similar logic to dict printing for vars(obj).items()

# --- Main execution ---
def main():
    try:
        print(f"Attempting to load pickle file: {FILENAME}")
        with open(FILENAME, 'rb') as f:
            # pickle.load reads the data from the file stream
            data = pickle.load(f)
        print(f"Successfully loaded data from {FILENAME}")
        print("-" * 50)
        print("Data Structure Analysis (Top Level):")
        print("-" * 50)
        print_structure(data, max_depth=MAX_DEPTH) # Start recursion from depth 0
        print("-" * 50)
        print("Analysis complete.")

    except FileNotFoundError:
        print(f"Error: File '{FILENAME}' not found.", file=sys.stderr)
        sys.exit(1)
    except ModuleNotFoundError as e:
         print(f"\nError: Could not find module '{e.name}'.", file=sys.stderr)
         print("This often happens if the pickle file contains objects (like custom classes or", file=sys.stderr)
         print("functions) from a library or code that isn't installed or available", file=sys.stderr)
         print("in your current Python environment.", file=sys.stderr)
         sys.exit(1)
    except pickle.UnpicklingError as e:
        print(f"Error unpickling file '{FILENAME}': {e}", file=sys.stderr)
        print("The file might be corrupted, incomplete, created with an incompatible", file=sys.stderr)
        print("Python version, or require specific classes/modules not available.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading or analyzing '{FILENAME}':", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        sys.exit(1)

if __name__ == "__main__":
    main()