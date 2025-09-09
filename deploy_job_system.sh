#!/bin/bash

# Deploy Job System to Render
# ===========================
# 
# This script helps you deploy the updated system with job endpoints

echo "🚀 Deploying Mineral Rights Job System to Render"
echo "================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    echo "Please run this script from your project root directory"
    exit 1
fi

# Check if all required files exist
echo "📋 Checking required files..."

required_files=(
    "api/app.py"
    "job_manager.py"
    "job_api_endpoints.py"
    "render_jobs_solution.py"
    "render.yaml"
    "requirements-clean.txt"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please ensure all files are present before deploying"
    exit 1
fi

echo "✅ All required files present"

# Check git status
echo ""
echo "📊 Git status:"
git status --porcelain

# Ask for confirmation
echo ""
read -p "🤔 Do you want to commit and push these changes? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Commit changes
echo ""
echo "📝 Committing changes..."
git add .
git commit -m "Add Render Jobs support for long-running processing

- Integrate job endpoints into existing API
- Add job management system for 8+ hour processing
- Create job monitoring and result retrieval
- Add frontend integration for job system
- Fix SSL/TLS connection issues with job endpoints"

# Push to GitHub
echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Changes pushed to GitHub!"
echo ""
echo "📋 Next Steps:"
echo "1. Go to Render Dashboard: https://dashboard.render.com"
echo "2. Your web service should automatically redeploy"
echo "3. Test the job system: curl https://your-app.onrender.com/test-jobs"
echo "4. Update your frontend to use the new job endpoints"
echo ""
echo "🔧 To test locally first:"
echo "   python test_render_jobs.py"
echo ""
echo "🌐 To test the API:"
echo "   curl https://your-app.onrender.com/health"
echo "   curl https://your-app.onrender.com/test-jobs"
echo ""
echo "📖 For detailed instructions, see: RENDER_JOBS_SETUP_GUIDE.md"
