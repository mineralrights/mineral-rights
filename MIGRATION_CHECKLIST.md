# Migration Checklist: Moving to New Accounts

This checklist follows the migration plan step-by-step. Check off items as you complete them.

## Prerequisites

- [ ] New Vercel account created and accessible
- [ ] New Google Cloud account created and accessible
- [ ] New Anthropic account created and accessible
- [ ] `gcloud` CLI installed and authenticated
- [ ] Access to GitHub repository

## Phase 1: Google Cloud Platform Setup

### 1.1 Create New GCP Project
- [ ] Run: `./scripts/setup_new_gcp_project.sh NEW_PROJECT_ID NEW_BUCKET_NAME`
- [ ] Or manually:
  - [ ] Create project: `gcloud projects create NEW_PROJECT_ID --name="Mineral Rights App"`
  - [ ] Set active: `gcloud config set project NEW_PROJECT_ID`
  - [ ] Enable billing in Cloud Console

### 1.2 Enable Required APIs
- [ ] APIs enabled (handled by setup script)
- [ ] Verify all APIs are enabled in Cloud Console

### 1.3 Create Service Account
- [ ] Service account created (handled by setup script)
- [ ] Roles granted (handled by setup script)
- [ ] Key file downloaded: `service-account-NEW_PROJECT_ID.json`
- [ ] Base64 encoded: `service-account-NEW_PROJECT_ID.base64.txt`

### 1.4 Create GCS Bucket
- [ ] Bucket created (handled by setup script)
- [ ] CORS configured (run `fix_gcs_cors.sh` with new bucket name)
- [ ] Permissions verified

### 1.5 Create Document AI Processor (OPTIONAL)
- [ ] **Note**: Document AI is OPTIONAL - the app works fine without it
- [ ] If you want to use Document AI:
  - [ ] Go to Cloud Console → Document AI → Processors
  - [ ] Create new processor (or use existing)
  - [ ] Note processor ID: `projects/PROJECT_NUMBER/locations/us/processors/PROCESSOR_ID`
  - [ ] Document endpoint format
- [ ] If skipping: App will use fallback splitting methods (simple page-based)

### 1.6 Create Secret Manager Secret
- [ ] Secret created (handled by setup script or manually)
- [ ] Anthropic API key added to secret
- [ ] Service account granted access

## Phase 2: Update Configuration Files

### 2.1 Update `cloudbuild.yaml`
- [ ] Create `migration_config.env` from template (run migration script first time)
- [ ] Fill in all values in `migration_config.env`
- [ ] Run: `./scripts/migrate_accounts.sh`
- [ ] Manually update `GOOGLE_CREDENTIALS_BASE64` in `cloudbuild.yaml` (lines 30 and 59)
- [ ] Verify bucket name updated
- [ ] Verify Document AI endpoint updated (if using Document AI, otherwise skip)

### 2.2 Update Frontend Environment Variables
- [ ] `web/vercel.json` updated (handled by migration script)
- [ ] `web/src/components/ModelManager.tsx` updated (handled by migration script)
- [ ] `web/src/lib/api_async.ts` updated (handled by migration script)
- [ ] `web/src/app/page.tsx` updated (handled by migration script)
- [ ] `web/src/lib/direct_gcs_upload.ts` updated (handled by migration script)
- [ ] Verify all API URL fallbacks point to new backend

### 2.3 Update Backend Code
- [ ] Check `api/app.py` for hardcoded values (should use env vars)
- [ ] Check `src/mineral_rights/large_pdf_processor.py` line 311 (should use env var)

## Phase 3: Vercel Migration

### 3.1 Create New Vercel Project
- [ ] Sign in to new Vercel account
- [ ] Import GitHub repository
- [ ] Create new project from repository

### 3.2 Configure Vercel Environment Variables
- [ ] Go to Project Settings → Environment Variables
- [ ] Set `NEXT_PUBLIC_API_URL` to new Cloud Run URL
- [ ] Verify environment variable is set

### 3.3 Update Vercel Configuration
- [ ] `web/vercel.json` already updated (from Phase 2)
- [ ] Verify configuration is correct

### 3.4 Deploy Frontend
- [ ] Push changes to trigger deployment
- [ ] Or manually deploy from Vercel dashboard
- [ ] Verify deployment URL works
- [ ] Test frontend loads correctly

## Phase 4: Anthropic Account Migration

### 4.1 Get New API Key
- [ ] Sign in to new Anthropic account
- [ ] Generate new API key from Console
- [ ] Copy the key (starts with `sk-ant-api03-`)

### 4.2 Update Secret Manager
- [ ] Update secret in Google Secret Manager:
  - `gcloud secrets versions add anthropic-api-key --data-file=-` (paste new key)
- [ ] Or via console: Secret Manager → `anthropic-api-key` → Add New Version
- [ ] Verify secret version is latest

### 4.3 Verify API Key Access
- [ ] Test API key works (use `/test-anthropic` endpoint after deployment)
- [ ] Or test manually: `curl https://api.anthropic.com/v1/messages ...`

## Phase 5: Deploy Backend

### 5.1 Authenticate Cloud Build
- [ ] Verify Cloud Build service account has permissions:
  - [ ] `roles/run.admin`
  - [ ] `roles/iam.serviceAccountUser`
  - [ ] `roles/storage.admin`
- [ ] Grant permissions if needed

### 5.2 Trigger Deployment
- [ ] Run: `gcloud builds submit --config cloudbuild.yaml .`
- [ ] Monitor build progress
- [ ] Wait for deployment to complete

### 5.3 Verify Deployment
- [ ] Check Cloud Run service: `gcloud run services describe mineral-rights-api --region=us-central1`
- [ ] Test endpoint: `curl https://mineral-rights-api-NEW_PROJECT_NUMBER.us-central1.run.app/health`
- [ ] Verify environment variables are set correctly
- [ ] Verify secret is accessible
- [ ] Check logs for any errors

## Phase 6: Testing & Validation

### 6.1 Test Frontend-Backend Connection
- [ ] Open Vercel deployment URL
- [ ] Check browser console for errors
- [ ] Verify `NEXT_PUBLIC_API_URL` is correct
- [ ] Test API connectivity

### 6.2 Test API Endpoints
- [ ] `/health` - Should return 200
- [ ] `/test-anthropic` - Should connect to Anthropic API
- [ ] `/predict` - Upload test PDF, verify processing works
- [ ] `/get-model-name` - Should return current model
- [ ] `/update-model-name` - Test model update

### 6.3 Test Full Workflow
- [ ] Upload PDF via frontend
- [ ] Verify processing completes
- [ ] Check results display correctly
- [ ] Verify CSV export works
- [ ] Test with different processing modes:
  - [ ] Single deed mode
  - [ ] Multi-deed mode
  - [ ] Page-by-page mode

## Phase 7: Cleanup (Optional)

### 7.1 Old Account Cleanup
- [ ] Verify new deployment works for 24-48 hours
- [ ] Delete old Vercel project (after confirming new one works)
- [ ] Delete old GCP project (or keep for reference)
- [ ] Revoke old Anthropic API key (after migration confirmed)

## Troubleshooting

### Common Issues

**Issue**: Cloud Build fails with permission errors
- **Solution**: Grant Cloud Build service account the required roles

**Issue**: Frontend can't connect to backend
- **Solution**: Check `NEXT_PUBLIC_API_URL` in Vercel environment variables

**Issue**: Secret Manager access denied
- **Solution**: Verify service account has `roles/secretmanager.secretAccessor`

**Issue**: GCS bucket access denied
- **Solution**: Verify service account has `roles/storage.objectAdmin` on bucket

**Issue**: Document AI processor not found
- **Solution**: Verify processor ID and endpoint format are correct

## Files Modified

- [ ] `cloudbuild.yaml`
- [ ] `web/vercel.json`
- [ ] `web/src/components/ModelManager.tsx`
- [ ] `web/src/lib/api_async.ts`
- [ ] `web/src/app/page.tsx`
- [ ] `web/src/lib/direct_gcs_upload.ts`
- [ ] `migration_config.env` (created)

## Environment Variables Summary

### Google Cloud Run (API Service)
- [ ] `ANTHROPIC_API_KEY` - From Secret Manager ✓
- [ ] `CLAUDE_MODEL_NAME` - Default: `claude-3-5-haiku-20241022` ✓
- [ ] `GCS_BUCKET_NAME` - New bucket name ✓
- [ ] `GOOGLE_CLOUD_PROJECT` - New project ID ✓
- [ ] `GOOGLE_CLOUD_LOCATION` - `us-central1` ✓
- [ ] `TASK_QUEUE_NAME` - `mineral-rights-queue` ✓
- [ ] `GOOGLE_CREDENTIALS_BASE64` - Base64-encoded service account JSON ✓
- [ ] `DOCUMENT_AI_ENDPOINT` - New processor endpoint (for worker) ✓

### Vercel (Frontend)
- [ ] `NEXT_PUBLIC_API_URL` - New Cloud Run URL ✓

## Notes

- Keep old accounts active until migration is fully verified
- Test thoroughly before deleting old resources
- Document any custom configurations or workarounds

