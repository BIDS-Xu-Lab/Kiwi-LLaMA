from os import environ

from dotenv import load_dotenv
import keyring

# This loads values from a .env file into the os.environ
load_dotenv()


def set_to_global(name: str, value, val_type: type):
    try:
        globals()[name] = val_type(value)
    except TypeError as exc:
        raise TypeError(f"Type for {name} must be of type {val_type.__name__}") from exc
    except ValueError as exc:
        raise ValueError(
            f"Value for {name} had some error.  You should not see this message."
        ) from exc
    except Exception as exc:
        raise Exception(
            f"General error in setting {name}.  Something has gone wrong and you should investigate."
        ) from exc


def get_env_var(name: str, val_type: type, default=None):
    val = environ.get(name, default)
    if val is None:
        if default is None:
            raise ValueError(f"Environment variable {name} is required")
        else:
            val = default
    set_to_global(name, val, val_type)


def get_secret_var(
    var_name: str, store_name: str, item_name: str, val_type: type, default=None
):
    """
    This is a helper function to get a secret from a secret store, similar to `get_env_var()`.

    To understand better, here's an example from the CLI:
        Set:
            `python -m keyring -b keyring.backends.SecretService.Keyring set Passwords pythonkeyringcli` -> asks for password
        Get:
            `python -m keyring -b keyring.backends.SecretService.Keyring get Passwords pythonkeyringcli` -> prints password

    And in your passwords manager, in my case Wallet Manager, this appears under the tree as
    "Secret Service" -> "Passwords" -> "Password for 'pythonkeyringcli' on 'Passwords'"

    But you'll notice that you can't retrieve all your other passwords.  This is a security
    feature.  Only passwords a program has set can be retrieved by that program.  However,
    this can become finicky.  This is a trade-off.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! NOTICE: You may have to use `sey_keyring()` if the correct backend is not automatically set. !!!
    !!! Please refer to https://pypi.org/project/keyring/ for more information.                      !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    val = keyring.get_password(store_name, item_name)
    if val is None:
        if default is None:
            raise ValueError(f"Secret {item_name} in store {store_name} is required")
        else:
            val = default

    set_to_global(var_name, val, val_type)


get_env_var("PROMPT_BATCH_SIZE", int, 100)

get_env_var("KIWI_LLAMA_MODEL_DIR", str, ".")

get_env_var("CUDA_VISIBLE_DEVICES", str, "0,1")

get_env_var("TRAINING_OUTPUT_DIR", str, "models/")

# This must be set by the user.
get_secret_var("HUGGING_FACE_API_TOKEN", "Passwords", "api_token", str)
