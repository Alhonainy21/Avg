#!/bin/bash                                                                                                                                      
MEM_LOGFILE="memory_usage_gb.csv"                                                                                                                
SERVER_LOGFILE="log_cache_2cli_lung3_server.txt"                                                                                                 
# Add headers to the memory usage CSV file                                                                                                       
echo "Timestamp,Available_Memory_GB" > "$MEM_LOGFILE"                                   
# Loop to log memory usage and upload logs to different folders every 30                
# seconds                                                                               
while true; do                                                                          
    ### Memory Logging Section ###                                                      
    # Capture the timestamp (YYYY-MM-DD HH:MM:SS)                                       
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")                                              
                                                                                        
                                                                                        
    # Capture available memory in kB from /proc/meminfo                                 
    AVAILABLE_MEM_KB=$(grep "MemAvailable" /proc/meminfo | awk '{print $2}')            
                                                                                        
                                                                                        
    # Convert the memory from kB to GB (1 GB = 1,048,576 kB)                            
    AVAILABLE_MEM_GB=$(echo "scale=2; $AVAILABLE_MEM_KB / 1048576" | bc)                
    echo "$TIMESTAMP,$AVAILABLE_MEM_GB" >> "$MEM_LOGFILE"                               
                                                                                        
                                                                                        
    ### Google Drive Sync Section ###                                                   
    # Sync memory log file to Google Drive folder for memory logs                       
    rclone copy "$MEM_LOGFILE" gdrive:Chameleon/Memory_2cli --update                    
    # Sync server log file to Google Drive folder for server logs                       
    rclone copy "$SERVER_LOGFILE" gdrive:Chameleon/Logs_2cli --update                   
                                                                                        
                                                                                        
    # Wait for 30 seconds before the next iteration                                     
    sleep 30                                                                            
done
