
mkdir -p data

for YEAR in {2001..2012}; do
    echo ${YEAR}
    mkdir -p data/${YEAR}
    curl https://www.eia.gov/electricity/data/eia861/archive/zip/861_${YEAR}.zip --output data/${YEAR}/eia_861.zip
    unzip data/${YEAR}/eia_861.zip -d data
done

for YEAR in {2013..2019}; do
    echo ${YEAR}
    mkdir -p data/${YEAR}
    curl https://www.eia.gov/electricity/data/eia861/archive/zip/f861${YEAR}.zip --output data/${YEAR}/eia_861.zip
    unzip data/${YEAR}/eia_861.zip -d data/${YEAR}
done

mkdir -p data/all_DR
mv data/*/Demand* data/all_DR
mv data/*/DSM* data/all_DR
mv data/*/dsm* data/all_DR
