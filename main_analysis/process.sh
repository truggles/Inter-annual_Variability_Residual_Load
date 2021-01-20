


#for REGION in "ERCOT" "NYISO" "PJM"; do
#for REGION in "ERCOT" "NYISO"; do
#for REGION in "ERCOT" "NYISO" "PJM"; do
#for REGION in "ERCOT" "NYISO" "PJM"; do
#for REGION in "PJM"; do
#for REGION in "FR"; do
#for REGION in "PJM"; do
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    python run_main_analysis.py $REGION &
#done


#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for HOURS in 1 2 3 4 5 10 15 20 25 50 75 100 200; do
#        python run_main_analysis.py $REGION $HOURS &
#    done
#    sleep 24*13
#done

DATE="20210120v1"
#for REGION in "FR"; do
for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
    for METHOD in "NOM" "TMY" "PLUS1"; do
        python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} 10 &
        #python plotting.py ${REGION} ${DATE}${METHOD} ${METHOD} 10
    done
done
