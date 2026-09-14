"""Configuration package.

Modules import the settings object they require explicitly.  Keeping package
initialisation side-effect free also lets V3's cached-data jobs run without
loading the V2 environment or secrets.
"""
