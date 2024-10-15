#!/bin/bash                                                                                                                                                               
LOGFILE="memory_usage_gb_server.csv"                                                                                                                                      
# Add headers to the CSV file                                                                                                                                             
echo "Timestamp,Available_Memory_GB" > "$LOGFILE"                                                                                                                         
while true; do                                                                                                                                                            
    # Capture the timestamp                                                                                                                                               
    TIMESTAMP=$(date +"%H:%M:%S")                                                                                                                                         
    # Capture available memory in kB from /proc/meminfo                                                                                                                   
    AVAILABLE_MEM_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)                                                                                                     
                                                                                                                                                                          
    # Print for debugging                                                                                                                                                 
    #echo "Available memory (kB): $AVAILABLE_MEM_KB"                                                                                                                      
    # Convert the memory from kB to GB (1 GB = 1,048,576 kB)                                                                                                              
    AVAILABLE_MEM_GB=$(echo "scale=2; $AVAILABLE_MEM_KB / 1048576" | bc)                                                                                                  
                                                                                                                                                                          
    # Print for debugging                                                                                                                                                 
    #echo "Available memory (GB): $AVAILABLE_MEM_GB"                                                                                                                      
    # Write the timestamp and available memory in GB to the log file                                                                                                      
    echo "$TIMESTAMP,$AVAILABLE_MEM_GB" >> "$LOGFILE"                                                                                                                     
    # Wait for 30 seconds before the next measurement                                                                                                                     
    sleep 30                                                                                                                                                              
done
