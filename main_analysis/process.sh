


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

############################
### Sensitivity analyses ###
############################
### N Hours
DATE="20210122v3"
METHOD="DT"
YEARS=10
for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
    for HOURS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 50 75 100 200; do
        python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} ${HOURS} ${YEARS} "SENSITIVITY" &
    done
    sleep 600
done

### N Years
#DATE="20210122v2"
#METHOD="DT"
#HOURS=10
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for YEARS in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
#        python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} ${HOURS} ${YEARS} "SENSITIVITY" &
#    done
#done

# Detrending
#DATE="20210122v1"
#HOURS=10
#YEARS=10
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for METHOD in "NOM" "DT"; do
#        python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} ${HOURS} ${YEARS} &
#    done
#done

DATE="20210120v1"
#for REGION in "ERCOT"; do # "NYISO" "PJM" "FR"; do
#    for METHOD in "NOM"; do # "TMY" "PLUS1"; do
#        #python run_main_analysis.py ${REGION} ${DATE}${METHOD} ${METHOD} 10 &
#    done
#done



#DATE="20210122v1"
#HOURS=10
#YEARS=10
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    python plotting.py ${REGION} ${DATE} "DUMMY" ${HOURS} ${YEARS} "Jan22Detrend"
#done
