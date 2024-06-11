bash
Copy code
#!/bin/bash

# Check if the script name is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script-number>"
    echo "Example: $0 1    # Runs train1.py"
    echo "         $0 2    # Runs train2.py"
    echo "         $0 3    # Runs train3.py"
    exit 1
fi

# Determine which Python script to run based on the input argument
case $1 in
    1)
        echo "Running train1.py"
        python3 train1.py
        ;;
    2)
        echo "Running train2.py"
        python3 train2.py
        ;;
    3)
        echo "Running train3.py"
        python3 train3.py
        ;;
    *)
        echo "Invalid script number: $1"
        echo "Please enter 1, 2, or 3."
        exit 1
        ;;
esac

echo "Script execution completed."