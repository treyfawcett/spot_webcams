# The IP address you use to communicate with the robot
# This can be determined by running python3 -m bosdyn.client $ROBOT_IP self-ip
export SELF_IP=192.168.80.100

# Generate a guid/secret pair for development.
export CRED_FILE=~/spot_dev_creds.txt

# One-time set up of credentials.
export GUID=$(python3 -c 'import uuid; print(uuid.uuid4())')
export SECRET=$(python3 -c 'import uuid; print(uuid.uuid4())')
printf "$GUID\n$SECRET" > $CRED_FILE 