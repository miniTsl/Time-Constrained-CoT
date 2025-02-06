#!/usr/bin/env expect

# Set timeout to -1, meaning no timeout
set timeout -1

# Start the original script
spawn bash sh/phi_4gpu.sh

# Automatically input 'y' when prompted
expect {
    "Do you wish to run the custom code? \\\[y/N\\\]" {
        send "y\r"
        exp_continue
    }
    eof
}  