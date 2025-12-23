import os

import forgather.nb.notebooks as nb

from .utils import BaseCommand, should_use_absolute_paths, write_output_or_edit


def trefs_cmd(args):
    """List templates referenced by configuration."""
    cmd = BaseCommand(args)

    match args.format:
        case "md":
            use_absolute_paths = should_use_absolute_paths(args)
            md_content = nb.render_referenced_templates_tree(
                cmd.env,
                cmd.meta.config_path(args.config_template),
                use_absolute_paths=use_absolute_paths,
            )
            write_output_or_edit(args, md_content, ".md", "trefs")
        case "files":
            data = ""
            # Yields # tuple(level: int, name: str, path: str)
            for level, name, path in cmd.env.find_referenced_templates(
                cmd.meta.config_path(args.config_template)
            ):
                data += os.path.relpath(path) + "\n"
            write_output_or_edit(args, data, ".txt")
        case "tree":
            data = render_template_hierarchy_tree(
                cmd.env, cmd.meta.config_path(args.config_template)
            )
            write_output_or_edit(args, data, ".txt")
        case "dot":
            data = render_template_hierarchy_dot(
                cmd.env, cmd.meta.config_path(args.config_template)
            )
            write_output_or_edit(args, data, ".dot")
        case "svg":
            svg_data = render_template_hierarchy_svg(
                cmd.env, cmd.meta.config_path(args.config_template)
            )
            write_output_or_edit(args, svg_data, ".svg")
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def render_template_hierarchy_tree(environment, template_path):
    """Render template hierarchy as a clean tree structure"""

    template_data = list(environment.find_referenced_templates(template_path))

    output = []
    output.append(f"Template Hierarchy for {template_path}")
    output.append("=" * (len(f"Template Hierarchy for {template_path}")))
    output.append("")

    # Group by hierarchies (detect separate root-level trees)
    current_hierarchy = []
    last_level = -1

    for level, name, path in template_data:
        # Detect new hierarchy starting
        if level == 0 and last_level >= 0:
            # End previous hierarchy
            if current_hierarchy:
                output.extend(_render_hierarchy_tree(current_hierarchy))
                output.append("")
            current_hierarchy = []

        current_hierarchy.append((level, name, path))
        last_level = level

    # Render final hierarchy
    if current_hierarchy:
        output.extend(_render_hierarchy_tree(current_hierarchy))

    return "\n".join(output)


def _render_hierarchy_tree(hierarchy_data):
    """Render a single hierarchy as a tree"""
    output = []

    # Use proper tree characters
    for i, (level, name, path) in enumerate(hierarchy_data):
        # Determine tree characters based on position and level
        indent = "    " * level

        if level == 0:
            # Root level
            prefix = "📁 "
        else:
            # Child levels - use tree characters
            is_last_at_level = True
            # Check if this is the last item at this level
            for j in range(i + 1, len(hierarchy_data)):
                next_level, _, _ = hierarchy_data[j]
                if next_level == level:
                    is_last_at_level = False
                    break
                elif next_level < level:
                    break

            # Build prefix with proper tree characters
            prefix = ""
            for l in range(level):
                if l == level - 1:
                    prefix += "└── " if is_last_at_level else "├── "
                else:
                    # Check if we need a vertical line at this level
                    has_sibling_below = False
                    for j in range(i + 1, len(hierarchy_data)):
                        check_level, _, _ = hierarchy_data[j]
                        if check_level == l:
                            has_sibling_below = True
                            break
                        elif check_level < l:
                            break
                    prefix += "│   " if has_sibling_below else "    "

        # Add file type emoji and path info
        if "trainer" in name.lower():
            emoji = "🏃 "
        elif "model" in name.lower():
            emoji = "🧠 "
        elif "dataset" in name.lower():
            emoji = "📊 "
        elif "callback" in name.lower():
            emoji = "🔄 "
        elif name.endswith(".yaml"):
            emoji = "📄 "
        else:
            emoji = "📋 "

        # Show relative path if different from name
        path_info = ""
        if path != name and not path.endswith(name):
            rel_path = os.path.relpath(path)
            if len(rel_path) < 50:  # Only show if not too long
                path_info = f" → {rel_path}"

        output.append(f"{prefix}{emoji}{name}{path_info}")

    return output


def render_template_hierarchy_dot(environment, template_path):
    """Render template hierarchy as Graphviz DOT format"""

    # Get the raw dependency relationships instead of using hierarchy levels
    load_sequence, dependencies = environment.get_template_dependencies(template_path)

    output = []
    output.append("digraph TemplateHierarchy {")
    output.append("    rankdir=TB;")
    output.append("    node [shape=box, style=rounded];")
    output.append("    edge [color=gray60];")
    output.append("")

    # Define node styles for different template types
    styles = {
        "config": 'fillcolor=lightblue, style="rounded,filled"',
        "trainer": 'fillcolor=lightgreen, style="rounded,filled"',
        "model": 'fillcolor=lightyellow, style="rounded,filled"',
        "dataset": 'fillcolor=lightcoral, style="rounded,filled"',
        "callback": 'fillcolor=lightgray, style="rounded,filled"',
        "default": 'fillcolor=white, style="rounded,filled"',
    }

    # Add nodes with styles
    all_templates = set()
    for name, path in load_sequence:
        all_templates.add(name)

    for name in sorted(all_templates):
        # Determine node style based on name
        style = styles["default"]
        if "trainer" in name.lower():
            style = styles["trainer"]
        elif "model" in name.lower():
            style = styles["model"]
        elif "dataset" in name.lower():
            style = styles["dataset"]
        elif "callback" in name.lower():
            style = styles["callback"]
        elif "config" in name.lower() or name.endswith(".yaml"):
            style = styles["config"]

        # Clean node name for DOT format
        clean_name = name.replace(".", "_").replace("/", "_").replace("-", "_")
        display_name = name.replace('"', '\\"')

        output.append(f'    {clean_name} [label="{display_name}", {style}];')

    output.append("")

    # Add edges based on actual dependency relationships
    # The dependencies dict shows: parent -> [children] relationships
    # In DOT format, we want: parent -> child (parent depends on/includes child)
    for parent, children in dependencies.items():
        parent_clean = parent.replace(".", "_").replace("/", "_").replace("-", "_")
        for child in sorted(children):
            child_clean = child.replace(".", "_").replace("/", "_").replace("-", "_")
            output.append(f"    {parent_clean} -> {child_clean};")

    output.append("}")

    return "\n".join(output)


def render_template_hierarchy_svg(environment, template_path):
    """Render template hierarchy as SVG using Graphviz"""
    import shutil
    import subprocess
    import tempfile

    # Check if dot command is available
    if not shutil.which("dot"):
        raise Exception(
            "Graphviz 'dot' command not found. Please install Graphviz to use SVG output."
        )

    dot_content = render_template_hierarchy_dot(environment, template_path)

    try:
        # Use dot command to generate SVG
        result = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_content,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to generate SVG with Graphviz: {e.stderr}")
    except Exception as e:
        raise Exception(f"Error generating SVG: {e}")
