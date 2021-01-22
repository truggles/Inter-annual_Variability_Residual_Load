


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


DATE="20210121v2"
METHOD="NOM"
for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
    for HOURS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 50 75 100 200; do
        python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} $HOURS &
    done
    sleep 1000
done

DATE="20210120v1"
#for REGION in "ERCOT"; do # "NYISO" "PJM" "FR"; do
#    for METHOD in "NOM"; do # "TMY" "PLUS1"; do
#        #python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} 10 &
#    done
#done



#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    python plotting.py ${REGION} ${DATE} "DUMMY" 10 "today"
#done
