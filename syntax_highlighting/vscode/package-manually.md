# Manual VSCode Extension Packaging

Since `vsce` requires Node.js 20+ and you may have compatibility issues, here's how to manually create a VSIX package:

## Method 1: ZIP-based VSIX Creation

A VSIX file is essentially a ZIP file with a specific structure. You can create one manually:

1. Create the proper directory structure:
   ```bash
   mkdir -p forgather-syntax-1.0.0/extension
   cp -r * forgather-syntax-1.0.0/extension/
   ```

2. Create the manifest files in the root:
   ```bash
   cd forgather-syntax-1.0.0
   ```

3. Create `[Content_Types].xml`:
   ```xml
   <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
   <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
     <Default Extension="json" ContentType="application/json"/>
     <Default Extension="vsixmanifest" ContentType="text/xml"/>
   </Types>
   ```

4. Create `extension.vsixmanifest`:
   ```xml
   <?xml version="1.0" encoding="utf-8"?>
   <PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
     <Metadata>
       <Identity Id="forgather-syntax" Version="1.0.0" Language="en-US" Publisher="forgather"/>
       <DisplayName>Forgather Configuration Syntax</DisplayName>
       <Description>Syntax highlighting for Forgather configuration files</Description>
       <Categories>Programming Languages</Categories>
       <Tags>forgather,yaml,jinja2,configuration</Tags>
     </Metadata>
     <Installation>
       <InstallationTarget Id="Microsoft.VisualStudio.Code" Version="[1.74.0,)"/>
     </Installation>
     <Dependencies/>
     <Assets>
       <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/>
     </Assets>
   </PackageManifest>
   ```

5. Create the VSIX:
   ```bash
   zip -r ../forgather-syntax-1.0.0.vsix . -x "*.md"
   ```

## Method 2: Use Docker with Newer Node.js

If you have Docker available:

```bash
# Create a temporary Dockerfile
cat > Dockerfile << 'EOF'
FROM node:20-alpine
RUN npm install -g @vscode/vsce
WORKDIR /workspace
CMD ["vsce", "package"]
EOF

# Build and run
docker build -t vsce-builder .
docker run --rm -v $(pwd):/workspace vsce-builder
```

## Method 3: Direct Installation (Recommended)

For development and testing, the simplest approach is to skip packaging entirely and use the direct installation method described in the main README:

1. Copy the extension directory to VSCode extensions folder
2. Restart VSCode
3. Configure file associations as needed

This avoids all the packaging complexity while providing the same functionality.