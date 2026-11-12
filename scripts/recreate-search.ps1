# Recreates the AI Search service for the Order Ops Agent and re-wires its roles.
# Run from any PowerShell window after `az login`. Nothing here uses API keys.

# ---------- Settings ----------
$rg          = "rg-orderagent-dev-eus2"
$location    = "eastus"
$searchName  = "srch-orderagent-dev-ar"
$storageName = "storderagentdevar"
$foundryName = "aif-orderagent-dev-ar"

# Stop if an az command fails (az is an external program, so we check its exit code)
function Check($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at: $step" -ForegroundColor Red
        exit 1
    }
    Write-Host "OK: $step" -ForegroundColor Green
}

# ---------- 1. Delete the old service (asks first) ----------
$exists = az search service show -n $searchName -g $rg --query name -o tsv 2>$null
if ($exists) {
    $answer = Read-Host "Delete search service '$searchName'? This removes its index. Type YES to continue"
    if ($answer -ne "YES") { Write-Host "Cancelled."; exit 0 }
    az search service delete -n $searchName -g $rg --yes
    Check "Delete old search service"
    Write-Host "Waiting 60 seconds for the name and free slot to be released..."
    Start-Sleep -Seconds 60
}

# ---------- 2. Create the new service ----------
# Free tier, system-assigned identity, and role-based access turned on (keys or Entra ID)
az search service create `
    -n $searchName -g $rg -l $location `
    --sku free `
    --identity-type SystemAssigned `
    --auth-options aadOrApiKey `
    --aad-auth-failure-mode http401WithBearerChallenge `
    --output none
Check "Create search service (Free)"

# ---------- 3. Look up the IDs we need ----------
$searchPrincipal = az search service show -n $searchName -g $rg --query identity.principalId -o tsv
$searchId        = az search service show -n $searchName -g $rg --query id -o tsv
$storageId       = az storage account show -n $storageName -g $rg --query id -o tsv
$foundryId       = az cognitiveservices account show -n $foundryName -g $rg --query id -o tsv
$myId            = az ad signed-in-user show --query id -o tsv
Check "Look up resource and identity IDs"

# A brand-new identity can take a moment to appear everywhere in Entra ID
Write-Host "Waiting 30 seconds for the new identity to replicate..."
Start-Sleep -Seconds 30

# ---------- 4. Roles for the Search service's identity ----------
az role assignment create --assignee-object-id $searchPrincipal --assignee-principal-type ServicePrincipal `
    --role "Storage Blob Data Reader" --scope $storageId --output none
Check "Search identity -> Storage Blob Data Reader on storage"

az role assignment create --assignee-object-id $searchPrincipal --assignee-principal-type ServicePrincipal `
    --role "Cognitive Services OpenAI User" --scope $foundryId --output none
Check "Search identity -> Cognitive Services OpenAI User on Foundry"

# ---------- 5. Role for you (so your local Function can query the index) ----------
az role assignment create --assignee-object-id $myId --assignee-principal-type User `
    --role "Search Index Data Reader" --scope $searchId --output none
Check "You -> Search Index Data Reader on search"

Write-Host ""
Write-Host "Done. Next: run the Import data (new) wizard in the portal." -ForegroundColor Cyan