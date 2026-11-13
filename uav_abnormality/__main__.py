from .cli import main
import sys

try:
    main()
except (ValueError, FileNotFoundError, FileExistsError) as exc:
    print("error: " + str(exc), file=sys.stderr)
    sys.exit(2)
