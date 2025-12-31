#!/usr/bin/env python
"""
Verify GitHub upload was successful
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

print("=" * 70)
print("GITHUB UPLOAD VERIFICATION")
print("=" * 70)

# 1. Check git status
print("\n1. Checking git status...")
returncode, stdout, stderr = run_command("git status")
if "nothing to commit" in stdout:
    print("   ✅ Working directory clean")
else:
    print("   ⚠ Uncommitted changes:")
    print(stdout)

# 2. Check remote
print("\n2. Checking remote repository...")
returncode, stdout, stderr = run_command("git remote -v")
print(f"   Remote: {stdout.strip()}")

# 3. Check commits
print("\n3. Checking commits...")
returncode, stdout, stderr = run_command("git log --oneline -3")
print(f"   Latest commits:\n{stdout}")

# 4. Check tags
print("\n4. Checking tags...")
returncode, stdout, stderr = run_command("git tag -l")
tags = [t.strip() for t in stdout.split('\n') if t.strip()]
print(f"   Tags: {', '.join(tags)}")

# 5. Test connection to GitHub
print("\n5. Testing GitHub connection...")
returncode, stdout, stderr = run_command("git fetch origin")
if returncode == 0:
    print("   ✅ Successfully connected to GitHub")
    
    # Compare local and remote
    returncode, local_commit, _ = run_command("git rev-parse main")
    returncode, remote_commit, _ = run_command("git rev-parse origin/main")
    
    if local_commit.strip() == remote_commit.strip():
        print("   ✅ Local and remote are synchronized")
    else:
        print("   ⚠ Local and remote differ")
        print(f"   Local:  {local_commit.strip()}")
        print(f"   Remote: {remote_commit.strip()}")
else:
    print("   ❌ Could not connect to GitHub")
    print(f"   Error: {stderr}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n📦 Package: geospatial-vit v1.0.0")
print(f"🌐 GitHub: https://github.com/jamesenglis/geospatial-vit-drone")
print(f"📥 Clone: git clone https://github.com/jamesenglis/geospatial-vit-drone.git")
print(f"⚡ Install: pip install -e .")

print("\n✅ Upload verification complete!")
print("=" * 70)
