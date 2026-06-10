"""THEMIS CLI - Rich-powered terminal interface.

Commands:
    themis ask "question"   - Single-shot legal Q&A
    themis chat             - Interactive multi-turn session
    themis eval             - Run the evaluation harness
    themis info             - Model metadata and training stats
    themis version          - Version info
"""

import sys
import io

# Force UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


app = typer.Typer(
    name="themis",
    help="THEMIS - Indian Legal Intelligence Engine",
    add_completion=False,
)
console = Console()

VERSION = "1.0.0"
MODEL_NAME = "Daniel2503/themis-mistral-7b-lora"

ASCII_ART = r"""
████████╗██╗  ██╗███████╗███╗   ███╗██╗███████╗
╚══██╔══╝██║  ██║██╔════╝████╗ ████║██║██╔════╝
   ██║   ███████║█████╗  ██╔████╔██║██║███████╗
   ██║   ██╔══██║██╔══╝  ██║╚██╔╝██║██║╚════██║
   ██║   ██║  ██║███████╗██║ ╚═╝ ██║██║███████║
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚══════╝"""


def display_banner():
    """Display the THEMIS ASCII art banner."""
    console.print()
    console.print(ASCII_ART, style="bold blue")
    console.print()
    console.print(
        f"[bold white]THEMIS[/bold white] v{VERSION}  "
        f"[dim]| Model: {MODEL_NAME}[/dim]"
    )
    console.print(
        "[dim]Domain: BNS 2023 | BNSS 2023 | IPC | Consumer Law | RTI Act[/dim]"
    )
    console.print()


# =============================================================================
# Commands
# =============================================================================


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your legal question"),
):
    """Ask a legal question and get a single-shot response."""
    from .infer import load_model, get_inference

    display_banner()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        try:
            load_model()
            inference = get_inference()
            progress.update(task, description="THEMIS is thinking...")
            result = inference.generate(question)
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        finally:
            progress.stop()

    # Display response in a Rich panel
    console.print()
    console.print(
        Panel(
            result.response,
            title="[bold blue]LEGAL ORIENTATION[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Footer
    console.print(
        f"[dim]Tokens: {result.input_tokens} in / {result.output_tokens} out | "
        f"Device: {result.device}[/dim]"
    )


@app.command()
def chat():
    """Start an interactive multi-turn legal Q&A session."""
    from .infer import load_model, get_inference

    display_banner()

    console.print("[bold green]Interactive Mode[/bold green]")
    console.print("Type your legal question, or 'exit' to quit.\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading model...", total=None)
            load_model()
            progress.stop()
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(1)

    inference = get_inference()
    history = []

    while True:
        try:
            question = console.input(
                "[bold cyan]Ask a legal question (or 'exit' to quit):[/bold cyan] "
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if question.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not question.strip():
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("THEMIS is thinking...", total=None)
            result = inference.generate(question, history)
            progress.stop()

        # Display response in a Rich panel
        console.print(
            Panel(
                result.response,
                title="[bold blue]LEGAL ORIENTATION[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Update history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.response})

        # Keep history manageable (last 10 exchanges)
        if len(history) > 20:
            history = history[-20:]

        console.print()


@app.command()
def eval(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed results"),
):
    """Run the evaluation harness on the held-out test set."""
    import json
    from .config import config

    display_banner()
    console.print("[bold yellow]Evaluation Mode[/bold yellow]\n")

    eval_file = config.eval_dir / "eval_set.json"
    if not eval_file.exists():
        console.print(f"[red]Eval set not found:[/red] {eval_file}")
        console.print("Run the data preprocessing pipeline first.")
        raise typer.Exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    console.print(f"Loaded {len(eval_set)} evaluation questions\n")

    try:
        from .infer import load_model, get_inference
        load_model()
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(1)

    inference = get_inference()
    results = []

    for i, item in enumerate(eval_set):
        question = item.get("instruction", "")
        console.print(f"[cyan]Q{i+1}:[/cyan] {question[:80]}...")

        try:
            result = inference.generate(question)
            results.append({
                "question": question,
                "expected": item.get("output", ""),
                "predicted": result.response,
            })
            if verbose:
                console.print(f"  [green]A:[/green] {result.response[:100]}...\n")
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}\n")

    # Save results
    output_file = config.eval_dir / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Evaluated {len(results)} questions[/green]")
    console.print(f"Results saved to {output_file}")


@app.command()
def info():
    """Display model metadata and training stats."""
    from rich.table import Table

    display_banner()

    table = Table(title="Model Information", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Base Model", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    table.add_row("LoRA Adapter", "Daniel2503/themis-mistral-7b-lora")
    table.add_row("LoRA Config", "rank=8, alpha=16, targets=[q_proj, v_proj]")
    table.add_row("Fine-tuning", "Unsloth + LoRA, 3 epochs")
    table.add_row("Training Data", "1,939 Indian legal Q&A pairs")
    table.add_row("Domain", "BNS 2023 | BNSS 2023 | IPC | Consumer Law | RTI Act")
    table.add_row("Quantization", "4-bit NF5 (bitsandbytes)")
    table.add_row("License", "MIT")

    console.print(table)


@app.command()
def version():
    """Show version and config."""
    console.print(f"THEMIS v{VERSION}")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Python: {sys.version.split()[0]}")


if __name__ == "__main__":
    app()
