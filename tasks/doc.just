# run doc tasks
[group('doc')]
doc: doc-build

# build documentation
[group('doc')]
doc-build format="numpy" output="docs": clean-docs
    uv run pdoc --docformat={{format}} --output-directory={{output}} {{SOURCES}}/{{PACKAGE}}

# serve documentation
[group('doc')]
doc-serve format="numpy" port="8088":
    uv run pdoc --docformat={{format}} --port={{port}} {{SOURCES}}/{{PACKAGE}}
