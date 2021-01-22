# Scrape data from http://mis.nyiso.com/public/P-58Clist.htm
# NYISO hourly actual load (demand) calculated as net gen - transmission (same as EIA)

for YEAR in {2002..2019}; do
    for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
        curl mis.nyiso.com/public/csv/palIntegrated/${YEAR}${MONTH}01palIntegrated_csv.zip --output nyiso_${YEAR}${MONTH}.zip
        mkdir nyiso_${YEAR}${MONTH}
        unzip nyiso_${YEAR}${MONTH}.zip -d nyiso_${YEAR}${MONTH}
    done
done
