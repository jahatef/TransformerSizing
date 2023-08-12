$Env:CONDA_EXE = "/fsx/quentin/jacob/gpt-neox-stuff/GEMMs_project/transformer_sizing/experiments/scripts/miniconda3-plot/bin/conda"
$Env:_CE_M = ""
$Env:_CE_CONDA = ""
$Env:_CONDA_ROOT = "/fsx/quentin/jacob/gpt-neox-stuff/GEMMs_project/transformer_sizing/experiments/scripts/miniconda3-plot"
$Env:_CONDA_EXE = "/fsx/quentin/jacob/gpt-neox-stuff/GEMMs_project/transformer_sizing/experiments/scripts/miniconda3-plot/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs