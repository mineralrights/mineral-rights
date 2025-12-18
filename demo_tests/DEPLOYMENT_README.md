# 🚀 Deploying Mineral Rights Analyzer

## Quick Deployment to Render (Recommended)

### Step 1: Prepare Your Code
1. Make sure all your files are committed to a Git repository (GitHub, GitLab, etc.)
2. Your repository should contain:
   - `app.py` (main Flask application)
   - `requirements.txt` (dependencies)
   - `render.yaml` (deployment configuration)
   - `templates/index.html` (web interface)
   - `document_classifier.py` (AI classifier)

### Step 2: Deploy to Render
1. **Sign up** at [render.com](https://render.com) (free account)
2. **Connect your GitHub** account
3. **Create New Web Service**
4. **Select your repository**
5. **Configure settings**:
   - **Name**: `mineral-rights-analyzer`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`

### Step 3: Set Environment Variables
In Render dashboard, add:
- **Key**: `ANTHROPIC_API_KEY`
- **Value**: Your Anthropic API key

### Step 4: Deploy!
- Click **"Create Web Service"**
- Wait 5-10 minutes for deployment
- Your app will be live at: `https://your-app-name.onrender.com`

## 🌐 Alternative Options

### Railway (Also Great)
1. Sign up at [railway.app](https://railway.app)
2. Connect GitHub repository
3. Add `ANTHROPIC_API_KEY` environment variable
4. Deploy automatically

### Heroku (More Expensive)
1. Install Heroku CLI
2. `heroku create your-app-name`
3. `heroku config:set ANTHROPIC_API_KEY=your_key`
4. `git push heroku main`

## 🔒 Security Notes
- Never commit your API key to Git
- Always use environment variables for secrets
- The app handles file uploads securely (temporary files only)

## 💰 Cost Estimates
- **Render**: Free for light usage
- **Railway**: Free tier, then $5/month
- **Heroku**: $7/month minimum

## 🛠 Troubleshooting
- **Build fails**: Check `requirements.txt` formatting
- **App crashes**: Check environment variables are set
- **Slow loading**: First request after inactivity takes longer (free tier limitation)

## 📱 Sharing Your App
Once deployed, share the URL with users:
- `https://your-app-name.onrender.com`
- Works on any device with internet
- No installation required for users 