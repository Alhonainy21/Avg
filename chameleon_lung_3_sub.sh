#!/bin/bash
#cd /
curl -L -o lung_3_sub.tar.gz https://www.dropbox.com/scl/fi/wokdzh5jy7nlp47g7k3q3/lung_3_sub.tar.gz?rlkey=c974yxicyq2ma8kovd2vfwk1i&st=586bnjfx&dl=0
tar -zxvf lung_3_sub.tar.gz
mv lung_3sub_folders lung_3_sub
cd lung_3_sub
#find name '._*' delete
find . -name '._*' -delete
cd test/lung_aca/aca_3                                                                                                                                                                   
mv *jpeg /lung_3_sub/test/lung_aca                                                                                                                                               
cd ..                                                                                                                                                                      
rm -r aca*                                                                                                                                                             
cd ..                                                                                                                                                                      
cd lung_n                                                                                                                                                                
cd n_3                                                                                                                                                               
mv *jpeg /lung_3_sub/test/lung_n                                                                                                                                     
cd ..                                                                                                                                                                      
rm -r n_*                                                                                                                                                                
cd .. 
cd lung_scc 
cd scc_3 
mv *jpeg /lung_3_sub/test/lung_scc      
cd ..     
rm -r scc* 
cd ../.. 
cd train                                                                                                                                                                                                                                                                                                                                           
cd lung_aca                                                                                                                                                               
cd aca_3                                                                                                                                                               
mv *jpeg /lung_3_sub/train/lung_aca                                                                                                                                               
cd ..                                                                                                                                                                      
rm -r aca*                                                                                                                                                               
cd ..                                                                                                                                                                      
cd lung_n                                                                                                                                                                
cd n_3                                                                                                                                                               
mv *jpeg /lung_3_sub/train/lung_n                                                                                                                                             
cd ..                                                                                                                                                                      
rm -r n_*                                                                                                                                                               
cd ..                                                                                                                                                                      
cd lung_scc                                                                                                                                                               
cd scc_3                                                                                                                                                            
mv *jpeg /lung_3_sub/train/lung_scc                                                                                                                                           
cd ..                                                                                                                                                                      
rm -r scc*
cd ../../.. 
echo "Done!!"
