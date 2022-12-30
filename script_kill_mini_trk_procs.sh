kill $(ps -aux | grep decision_tracker | awk '{ print $2 }')
