#!/bin/bash                                                                                                                                      
LOGFILE="memory_usage_gb.csv"                                                                                                                    
# Add headers to the CSV file                                                                                                                    
echo "Timestamp,Available_Memory_GB" > "$LOGFILE"                                                                                                
while true; do                                                                                                                                   
    # Capture the timestamp                                                                                                                      
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")                                                                                                       
                                                                                                                                                 
    # Capture available memory in kB from /proc/meminfo                                                                                          
    AVAILABLE_MEM_KB=$(grep "MemAvailable" /proc/meminfo | awk '{print $2}')                                                                     
                                                                                                                                                 
    # Convert the memory from kB to GB (1 GB = 1,048,576 kB)                                                                                     
    #AVAILABLE_MEM_GB=$(LC_NUMERIC="en_US.UTF-8" echo "scale=2; $AVAILABLE_MEM_KB / 1048576" | bc)                                               
    AVAILABLE_MEM_GB=$(echo "scale=2; $AVAILABLE_MEM_KB / 1048576" | bc)                                                                         
    # Write the timestamp and available memory in GB to the log file in                                                                          
    # CSV format                                                                                                                                 
    echo "$TIMESTAMP,$AVAILABLE_MEM_GB" >> "$LOGFILE"                                                                                            
                                                                                                                                                 
    # Wait for 30 seconds before the next measurement                                                                                            
    sleep 30                                                                                                                                     
done
