#!/bin/bash

# Submit CQ-500 Jobs
sbatch submit_job_downstream_cq500_BleedLocation-Left.sh
sbatch submit_job_downstream_cq500_BleedLocation-Right.sh
#sbatch submit_job_downstream_cq500_CalvarialFracture.sh
sbatch submit_job_downstream_cq500_ChronicBleed.sh
sbatch submit_job_downstream_cq500_MassEffect.sh
sbatch submit_job_downstream_cq500_MidlineShift.sh
# sbatch submit_job_downstream_cq500_Fracture.sh
# sbatch submit_job_downstream_cq500_OtherFracture.sh
sbatch submit_job_downstream_cq500_ICH.sh
sbatch submit_job_downstream_cq500_IPH.sh
sbatch submit_job_downstream_cq500_IVH.sh
sbatch submit_job_downstream_cq500_SDH.sh
sbatch submit_job_downstream_cq500_EDH.sh
sbatch submit_job_downstream_cq500_SAH.sh

# Submit NYU Jobs
# sbatch submit_job_downstream_nyu_cancer.sh
# sbatch submit_job_downstream_nyu_dementia.sh
# sbatch submit_job_downstream_nyu_edema.sh
# sbatch submit_job_downstream_nyu_hydrocephalus.sh
# sbatch submit_job_downstream_nyu_fracture.sh
# sbatch submit_job_downstream_nyu_EDH.sh
# sbatch submit_job_downstream_nyu_ICH.sh
# sbatch submit_job_downstream_nyu_IPH.sh
# sbatch submit_job_downstream_nyu_IVH.sh
# sbatch submit_job_downstream_nyu_SAH.sh
# sbatch submit_job_downstream_nyu_SDH.sh

# Submit RSNA Jobs
# sbatch submit_job_downstream_rsna_any.sh
# sbatch submit_job_downstream_rsna_intraparenchymal.sh
# sbatch submit_job_downstream_rsna_intraventricular.sh
# sbatch submit_job_downstream_rsna_subarachnoid.sh
# sbatch submit_job_downstream_rsna_subdural.sh
# sbatch submit_job_downstream_rsna_epidural.sh