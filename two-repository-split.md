# Guide: Two-Repository Git Split (Private & Public)

This guide walks you through setting up a workflow where your work-in-progress (WIP), drafts, and experimental code remain securely hidden in a **Private GitHub Repository**, while your polished, production-ready code is pushed to a **Public GitHub Repository**.

---

## Architecture Overview
* **Local Machine:** Where you write code. You will use distinct branches to separate private work from public releases.
* **`origin` (Private Repo):** Your backup and staging environment. All your local branches (including WIP) can be safely pushed here.
* **`upstream` (Public Repo):** Your public portfolio or shared ecosystem. Only fully completed, cleaned code from your `main` branch gets pushed here.

---

## Step 1: Initial Setup

### 1. Create the repositories on GitHub
1. Go to GitHub and create a new **Private** repository (e.g., `my-private-project`). Do **not** initialize it with a README, .gitignore, or license.
2. Create a second, new **Public** repository (e.g., `my-public-project`). Do **not** initialize it with any files.

### 2. Initialize your local project
Open your terminal on your computer and run the following commands:

```bash
# Create your local project folder
mkdir my-project
cd my-project

# Initialize a clean Git repository
git init -b main
```

### 3. Link both repositories (Remotes)
Link your private repository as `origin` and your public repository as `upstream`:

```bash
# Link the Private repository as 'origin'
git remote add origin https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git

# Link the Public repository as 'upstream'
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_PUBLIC_REPO.git

# Verify that both remotes are configured correctly
git remote -v
```

*Your terminal should output four lines: two for `origin` (fetch/push) and two for `upstream` (fetch/push).*

---

## Step 2: The Core Workflow

To keep your work hidden from public view, you must separate your day-to-day coding from your public releases using local branches.

### 1. Day-to-Day Private Development (WIP)
Never write experimental or unfinished code directly on your local `main` branch. Always use a feature branch.

```bash
# 1. Create and switch to a local feature branch
git checkout -b feature-wip

# or
git switch -c feature-wip

# 2. Work on your code, then stage and commit locally
git add .
git commit -m "Work in progress: building feature component"

# 3. Safely push to your private repository
# The '-u' flag links this local branch to 'origin' for future shortcuts
git push -u origin feature-wip
```
*Your work is now securely backed up on GitHub, but completely hidden from the public.*

*Some useful commands.*
```bash
git status
git branch
git checkout feature-wip
git checkout main

# unstage everything
git restore --staged .

# Undo commits but KEEP your code changes
git reset HEAD~N

# Undo commits and Keep changes staged (ready to commit again immediately)
git reset --soft HEAD~N

# break the link between the current local branch and its corresponding branch on GitHub
git branch --unset-upstream

# merge feature branch to main branch
# note: `git merge` is strictly offline except for when `git pull` is used because `git pull` is a short cut of two commands: `git fetch` and `git merge`
git switch main
git merge feature-branch
```

### 2. Launching Code to the Public Repo
When your feature is fully completed, tested, and ready for the world to see, bring it into your local `main` branch and deploy it publicly.

```bash
# 1. Switch back to your local main branch
git checkout main

# 2. Merge your completed feature into main
git merge feature-wip

# 3. Push the clean, finished product to your PUBLIC repository
git push upstream main
```

---

## Step 3: Keeping Everything Synced

If you make quick updates directly on the public GitHub interface (like editing a README) or accept public pull requests, you must sync your local machine and private repository.

```bash
# 1. Switch to your main branch
git checkout main

# 2. Pull down the latest changes from the public repo
git pull upstream main

# 3. Sync your private repository so it is up-to-date
git push origin main
```

---

## Safety Best Practices
1. **Always specify the remote name:** Avoid using the generic `git push` shortcut unless you are certain your branch tracker is correctly configured. Explicitly typing `git push origin <branch>` or `git push upstream main` prevents accidental leaks.
2. **Double check before pushing:** Use `git remote -v` if you ever forget which repository is public (`upstream`) or private (`origin`).
3. **Use `.gitignore`:** If there are files (like `.env` configuration files or local API keys) that should *never* go public, list them in a `.gitignore` file at the root of your project immediately.