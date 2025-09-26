#!/usr/bin/env python3
"""
Fix for original demo.py - Update to use optimized engine
"""

import shutil
import os

def backup_and_update_demo():
    """Backup original demo and update it to use optimized engine"""
    
    print("🔧 FIXING ORIGINAL DEMO.PY")
    print("=" * 40)
    
    # Backup original demo
    if os.path.exists('demo.py'):
        print("📁 Backing up original demo.py to demo_original.py...")
        shutil.copy('demo.py', 'demo_original.py')
        print("   ✓ Backup created")
    
    # Read original demo
    with open('demo.py', 'r') as f:
        content = f.read()
    
    # Replace imports and engine
    print("🔄 Updating demo.py to use optimized engine...")
    
    # Replace the import
    content = content.replace(
        'from core.evolution_engine import EvolutionEngine',
        'from core.evolution_engine_optimized import OptimizedEvolutionEngine'
    )
    
    # Replace the engine initialization
    content = content.replace(
        'evolution_engine = EvolutionEngine(',
        'evolution_engine = OptimizedEvolutionEngine('
    )
    
    # Update title
    content = content.replace(
        '🧬 Prokaryotic Genome Evolution Simulator - Demo',
        '🚀 Optimized Prokaryotic Genome Evolution Simulator - Demo'
    )
    
    # Add optimization note
    content = content.replace(
        'print("   ✓ Point mutation rate: 1e-6 (per bp per generation)")',
        'print("   ✓ Point mutation rate: 1e-6 (per bp per generation) - OPTIMIZED")'
    )
    
    # Write updated demo
    with open('demo.py', 'w') as f:
        f.write(content)
    
    print("   ✓ demo.py updated to use OptimizedEvolutionEngine")
    print("   ✓ Original version saved as demo_original.py")
    print()
    print("📊 Changes made:")
    print("   - Import changed to OptimizedEvolutionEngine")
    print("   - Engine initialization updated")
    print("   - Added optimization indicators")
    print()
    print("✅ demo.py is now optimized and should run much faster!")
    print("   Run 'python demo.py' to test the improved performance")

if __name__ == "__main__":
    backup_and_update_demo()