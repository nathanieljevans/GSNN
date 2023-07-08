#!/bin/sh

# Download datasets and document sources 
# author: nathaniel evans
# email: evansna@ohsu.edu 
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# LINCS L1000 Phase II - beta 

# LINCS info 
l1000_phaseII_geneinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
l1000_phaseII_cellineinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt

# Level 4 LINCS 
l1000_phaseII_lvl34_meta=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt

# Level 5 LINCS 
LINCS_LEVEL5_META=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt
LINCS_LEVEL5_CP=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx

# CLUE compound data
# info: https://clue.io/releases/data-dashboard
l1000_phaseII_compoundinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt

# reactome pathway genesets 
# info: https://reactome.org/download-data
reactome_genesets=https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# cell line info - ccle
# paper: nature.com/articles/s41586-019-1186-3
# info: https://depmap.org/portal/download/
# file name: sample_info.csv
# file size: ???
# readme: https://depmap.org/portal/download/?releasename=DepMap+Public+22Q1&filename=sample_info.csv
info=https://ndownloader.figshare.com/files/34008503
#old https://ndownloader.figshare.com/files/26261569

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# cell line mutation data - ccle 
# info: https://depmap.org/portal/download/
# Citation: Mahmoud Ghandi, Franklin W. Huang, Judit Jané-Valbuena, Gregory V. Kryukov, ... Todd R. Golub, Levi A. Garraway & William R. Sellers. 2019. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature 569, 503-508 (2019).
# file name: 
# file size: 
# readme: https://depmap.org/portal/download/?releasename=DepMap+Public+22Q1&filename=CCLE_mutations.csv
mut=https://ndownloader.figshare.com/files/34008434
# old: https://ndownloader.figshare.com/files/27902118

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# cell line CNV data - ccle 
# info: https://depmap.org/portal/download/
# Citation: Mahmoud Ghandi, Franklin W. Huang, Judit Jané-Valbuena, Gregory V. Kryukov, ... Todd R. Golub, Levi A. Garraway & William R. Sellers. 2019. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature 569, 503-508 (2019).
# file name: 
# file size:
# readme: https://depmap.org/portal/download/?releasename=DepMap+Public+22Q1&filename=CCLE_gene_cn.csv 
cnv=https://ndownloader.figshare.com/files/34008428
# old: https://ndownloader.figshare.com/files/27902124

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# RNA expression data - CCLE 
# paper: nature.com/articles/s41586-019-1186-3
# info: https://depmap.org/portal/download/
# file name: CCLE_expression.csv
# file size: ???
# readme: https://depmap.org/portal/download/?releasename=DepMap+Public+22Q1&filename=CCLE_expression.csv
rna_expr=https://ndownloader.figshare.com/files/34008404
# old: https://ndownloader.figshare.com/files/26261476
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# cell line methylation data - ccle (Release: CCLE 2019)
# info: https://depmap.org/portal/download/
# Citation: Mahmoud Ghandi, Franklin W. Huang, Judit Jané-Valbuena, Gregory V. Kryukov, ... Todd R. Golub, Levi A. Garraway & William R. Sellers. 2019. Next-generation characterization of the Cancer Cell Line Encyclopedia. Nature 569, 503-508 (2019).
# file name: CCLE_RRBS_TSS_CpG_clusters_20180614.txt
# file size: 
# readme: https://depmap.org/portal/download/?releasename=CCLE+2019&filename=CCLE_RRBS_TSS1kb_20181022.txt.gz
CpG_rrbs='https://depmap.org/portal/download/api/download?file_name=ccle%2FCCLE_RRBS_TSS_CpG_clusters_20180614.txt&bucket=depmap-external-downloads'


# Targetome 
# Blucher, A. S., Choonoo, G., Kulesz-Martin, M., Wu, G. & McWeeney, S. K. Evidence-Based Precision Oncology with the Cancer Targetome. 
# Trends Pharmacol. Sci. (2017). doi:10.1016/j.tips.2017.08.006
targetome=https://raw.githubusercontent.com/ablucher/The-Cancer-Targetome/master/results_070617/Targetome_FullEvidence_070617.txt



### BROAD PRISM data 
# Steven M Corsello, Rohith T Nagari, Ryan D Spangler, Jordan Rossen, Mustafa Kocak, Jordan G Bryan, Ranad Humeidi, David Peck, 
# Xiaoyun Wu, Andrew A Tang, Vickie MWang, Samantha A Bender, Evan Lemire, Rajiv Narayan, Philip Montgomery, Uri Ben-David, 
# Yejia Chen, Matthew G Rees, Nicholas J Lyons, James M McFarland, Bang TWong, Li Wang, Nancy Dumont, Patrick J O'Hearn, 
# Eric Stefan, John G Doench, Heidi Greulich, Matthew Meyerson, Francisca Vazquez, Aravind Subramanian, Jennifer A Roth, 
# Joshua A Bittker, Jesse S Boehm, Christopher C Mader, Aviad Tsherniak, Todd R Golub. 2019. 
# Non-oncology drugs are a source of previously unappreciated anti-cancer activity. bioRxiv doi: 10.1101/730119
prism_primary_lfc=https://ndownloader.figshare.com/files/20237709
prism_primary_sample_info=https://ndownloader.figshare.com/files/20237715
prism_primary_cell_info=https://ndownloader.figshare.com/files/20237718

prism_second_lfc=https://ndownloader.figshare.com/files/20237757
prism_second_sample_info=https://ndownloader.figshare.com/files/20237763
prism_second_cell_info=https://ndownloader.figshare.com/files/20237769


### NCI Almanac data 
# Holbeck, S. L., Camalier, R., Crowell, J. A., Govindharajulu, J. P., Hollingshead, M., Anderson, L. W., et al. (2017). 
# The National Cancer Institute ALMANAC: a comprehensive screening resource for the detection of anticancer drug pairs with 
# enhanced therapeutic activity. Cancer Res. 77, 3564–3576. doi: 10.1158/0008-5472.can-17-0489
nci_almanac="https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip?version=1&modificationDate=1510057275000&api=v2&download=true"
nsc2sid="https://wiki.nci.nih.gov/download/attachments/155844992/NSC_PubChemSID.csv?version=1&modificationDate=1378730186000&api=v2&download=true"

#######################################################################################################################
#######################################################################################################################


# check that dir/file exists and then make/download
ROOT=$1
[ ! -d "$ROOT" ] && mkdir $ROOT

# date of download 
[ -f "$ROOT/date_of_download.txt" ] && rm $ROOT/date_of_download.txt
date > $ROOT/date_of_download.txt

[ ! -f "$ROOT/ComboDrugGrowth_Nov2017.csv" ] && wget $nci_almanac -O $ROOT/ComboDrugGrowth_Nov2017.zip
[ -f "$ROOT/ComboDrugGrowth_Nov2017.zip" ] && unzip $ROOT/ComboDrugGrowth_Nov2017.zip -d $ROOT
[ -f "$ROOT/ComboDrugGrowth_Nov2017.zip" ] && rm $ROOT/ComboDrugGrowth_Nov2017.zip
[ ! -f "$ROOT/NSC_PubChemSID.csv" ] && wget $nsc2sid -O $ROOT/NSC_PubChemSID.csv

[ ! -f "$ROOT/primary-screen-replicate-collapsed-logfold-change.csv" ] && wget $prism_primary_lfc -O $ROOT/primary-screen-replicate-collapsed-logfold-change.csv
[ ! -f "$ROOT/primary-screen-replicate-collapsed-treatment-info.csv" ] && wget $prism_primary_sample_info -O $ROOT/primary-screen-replicate-collapsed-treatment-info.csv
[ ! -f "$ROOT/primary-screen-cell-line-info.csv" ] && wget $prism_primary_cell_info -O $ROOT/primary-screen-cell-line-info.csv

[ ! -f "$ROOT/secondary-screen-replicate-collapsed-logfold-change.csv" ] && wget $prism_second_lfc -O $ROOT/secondary-screen-replicate-collapsed-logfold-change.csv
[ ! -f "$ROOT/secondary-screen-replicate-collapsed-treatment-info.csv" ] && wget $prism_second_sample_info -O $ROOT/secondary-screen-replicate-collapsed-treatment-info.csv
[ ! -f "$ROOT/secondary-screen-cell-line-info.csv" ] && wget $prism_second_cell_info -O $ROOT/secondary-screen-cell-line-info.csv

[ ! -f "$ROOT/targetome.txt" ] && wget $targetome -O $ROOT/targetome.txt

[ ! -f "$ROOT/geneinfo_beta.txt" ] && wget $l1000_phaseII_geneinfo -O $ROOT/geneinfo_beta.txt
[ ! -f "$ROOT/cellinfo_beta.txt" ] && wget $l1000_phaseII_cellineinfo -O $ROOT/cellinfo_beta.txt

[ ! -f "$ROOT/compoundinfo_beta.txt" ] && wget $l1000_phaseII_compoundinfo -O $ROOT/compoundinfo_beta.txt
[ ! -f "$ROOT/Uniprot2Reactome.txt" ] && wget $reactome_genesets -O $ROOT/Uniprot2Reactome.txt

# LEVEL 4
[ ! -f "$ROOT/instinfo_beta.txt" ] && wget $l1000_phaseII_lvl34_meta -O $ROOT/instinfo_beta.txt  #  NOTE: Need this to map "ccle_name" (siginfo_beta doesn't have it)

#LEVEL 5 
[ ! -f "$ROOT/level5_beta_trt_cp_n720216x12328.gctx" ] && wget $LINCS_LEVEL5_CP -O $ROOT/level5_beta_trt_cp_n720216x12328.gctx
[ ! -f "$ROOT/siginfo_beta.txt" ] && wget $LINCS_LEVEL5_META -O $ROOT/siginfo_beta.txt

# OMIC data 
[ ! -f "$ROOT/ccle_expression.txt" ] && wget $rna_expr -O $ROOT/ccle_expression.txt
[ ! -f "$ROOT/ccle_info.txt" ] && wget $info -O $ROOT/ccle_info.txt
[ ! -f "$ROOT/ccle_mutation.txt" ] && wget $mut -O $ROOT/ccle_mutation.txt
[ ! -f "$ROOT/ccle_cnv.txt" ] && wget $cnv -O $ROOT/ccle_cnv.txt
[ ! -f "$ROOT/ccle_methyl.txt" ] && wget $CpG_rrbs -O $ROOT/ccle_methyl.txt --referer='https://depmap.org/portal/download/all/' --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"

echo 'downloads complete'