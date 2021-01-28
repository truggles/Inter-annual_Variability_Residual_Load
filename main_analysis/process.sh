

############################
### Sensitivity analyses ###
############################
### N Hours
#DATE="20210122v3"
#DEM_METHOD="DT"
#RES_METHOD="NOM"
#YEARS=10
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for HOURS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 50 75 100 200; do
#        python run_main_analysis.py ${REGION} ${DATE}${DEM_METHOD}${RES_METHOD} ${DEM_METHOD} ${RES_METHOD} ${HOURS} ${YEARS} "SENSITIVITY" &
#    done
#    sleep 600
#done

### N Years
#DATE="20210122v2"
#DEM_METHOD="DT"
#RES_METHOD="NOM"
#HOURS=10
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for YEARS in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
#        python run_main_analysis.py ${REGION} ${DATE}${DEM_METHOD}${RES_METHOD} ${DEM_METHOD} ${RES_METHOD} ${HOURS} ${YEARS} "SENSITIVITY" &
#    done
#done

# Detrending
#DATE="20210122v1"
#HOURS=10
#YEARS=10
#RES_METHOD="NOM"
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for DEM_METHOD in "NOM" "DT"; do
#        python run_main_analysis.py ${REGION} ${DATE}${DEM_METHOD}${RES_METHOD} ${DEM_METHOD} ${RES_METHOD} ${HOURS} ${YEARS} &
#    done
#done

##########################################
### Running climate and weather checks ###
##########################################
#DATE="20210127v2"
#HOURS=10
#YEARS=10
#DEM_METHOD="DT"
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for RES_METHOD in "NOM" "TMY" "PLUS1"; do
#        python run_main_analysis.py ${REGION} ${DATE}${DEM_METHOD}${RES_METHOD} ${DEM_METHOD} ${RES_METHOD} ${HOURS} ${YEARS} &
#    done
#done


###############################
### Running default methods ###
###############################
#DATE="20210127v1"
#HOURS=10
#YEARS=10
#DEM_METHOD="DT"
#RES_METHOD="NOM"
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    for HOURS in 1 5 10; do
#        python run_main_analysis.py ${REGION} ${DATE}${DEM_METHOD}${RES_METHOD} ${DEM_METHOD} ${RES_METHOD} ${HOURS} ${YEARS} &
#    done
#done



############################
### Plotting             ###
############################
DATE="20210127v1"
HOURS=10
YEARS=10
#for REGION in "ERCOT" "NYISO" "PJM" "FR"; do
#    python plotting.py ${REGION} ${DATE} "DUMMY" ${HOURS} ${YEARS} "${REGION}_Jan27ClimAndWeather"
for REGION in "ALL"; do
    python plotting.py ${REGION} ${DATE} "DUMMY" ${HOURS} ${YEARS} "Jan27"
done
