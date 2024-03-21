#!/bin/bash
# Run as `source connect_tiago.sh TIAGO_NUM ROBOT_IFACE`
# where:
#   - TIAGO_NUM: is the ID number of the robot you are connected
#   - ETH: just pass 1 to connect via ethernet or 0 via TIAGo_WIFI


ETH=$1
TIAGO_NUM=89
#echo $TIAGO_NUM

# Change ROS Master
if [ "$ETH" -eq 0 ]; then
    echo "Connecting through Ethernet"
    export ROS_MASTER=10.68.0.1
else
    export ROBOT_HOSTNAME=tiago-89c.network.uni
    export ROS_MASTER=`getent hosts ${ROBOT_HOSTNAME} | awk '{ print $1 }'`
fi
export ROS_MASTER_URI=http://${ROS_MASTER}:11311


# wait until the master is pingable
echo "Wait until ros master is reachable..."
ping_cancelled=false    # Keep track of whether the loop was cancelled, or succeeded
until ping -c1 ${ROS_MASTER} >/dev/null 2>&1; do :; done &
trap "kill $!; ping_cancelled=true" SIGINT
wait $!          # Wait for the loop to exit, one way or another
trap - SIGINT    # Remove the trap, now we're done with it
echo "Done pinging ros master."

# check the ros master is running
# rostopic list &>/dev/null
# RETVAL=$?
# if [ $RETVAL -ne 0 ]; then
#     echo "[ERROR] connection with ROS MASTER not enstablished"
# else
#     echo "[OK] Connected to ROS MASTER"
# fi

# Add forwarding address to DNS
#echo $ROBOT_IFACE
#THIS_IP=`ifconfig ${ROBOT_IFACE} | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`
THIS_IP=`ip route get ${ROS_MASTER} | grep "src" | sed 's/.*src \([0-9\.]*\).*/\1/'`
export ROS_HOSTNAME=${HOSTNAME}
export ROS_IP=${THIS_IP}
# add to bashrc
echo "export ROS_MASTER_URI=${ROS_MASTER_URI}" >> ~/.bashrc
echo "export ROS_HOSTNAME=${ROS_HOSTNAME}" >> ~/.bashrc
# get the last part of THIS_IP and set is as ROS_DOMAIN_ID
CN=`echo ${THIS_IP} | cut -d . -f 4`
export ROS_NAMESPACE=$CN
echo "export ROS_NAMESPACE=${ROS_NAMESPACE}" >> ~/.bashrc


echo "${ROS_MASTER} tiago-${TIAGO_NUM}c" >> /etc/hosts
echo "executing addLocalDns -u \"${ROS_HOSTNAME}\" -i \"${ROS_IP}\""
sshpass -p "palroot" ssh root@${ROS_MASTER} "addLocalDns -u \"${ROS_HOSTNAME}\" -i \"${ROS_IP}\""

# launch viz software
source /opt/ros/noetic/setup.bash
roslaunch plotjuggler_ros plotjuggler.launch