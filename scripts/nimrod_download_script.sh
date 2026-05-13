#! /usr/bin/sh

wget -e --robots=off --mirror --no-parent -np -r https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/$1/ --header "Authorization: Bearer $CEDA_TOKEN"
