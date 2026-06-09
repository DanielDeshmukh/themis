"""THEMIS CLI - Rich-powered terminal interface.

Commands:
    themis ask "question"    - Single-shot legal Q&A
    themis chat              - Interactive multi-turn session
    themis eval              - Run evaluation harness
    themis info              - Model metadata
    themis version           - Version info
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="themis",
    help="THEMIS - Indian Legal Intelligence Engine",
    add_completion=False,
)
console = Console()

VERSION = "1.0.0"
MODEL_NAME = "danieldeshmukh/themis-mistral-7b-lora"


def display_banner():
    """Display the THEMIS banner."""
    banner = Text()
    banner.append("THEMIS", style="bold blue")
    banner.append(" - Indian Legal Intelligence Engine v", style="white")
    banner.append(VERSION, style="green")

    info = Text()
    info.append(f"Model: {MODEL_NAME}\n", style="dim")
    info.append("Domain: BNS 2023 | BNSS 2023 | IPC | Consumer Law", style="dim")

    console.print(Panel.fit(
        banner,
        subtitle=info,
        border_style="blue",
    ))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your legal question"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Ask a legal question and get a response."""
    from infer import load_model, get_inference

    display_banner()
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("THEMIS is thinking...", total=None)

        try:
            load_model()
            inference = get_inference()
            result = inference.generate(question)
        except Exception as e:
            progress.stop()
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        finally:
            progress.stop()

    # Display response
    console.print(Panel(
        result.response,
        title="[bold blue]LEGAL ORIENTATION[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))

    # Footer info
    console.print(
        f"\n[dim]Tokens: {result.input_tokens} in / {result.output_tokens} out | "
        f"Device: {result.device}[/dim]"
    )


@app.command()
def chat():
    """Start an interactive multi-turn legal Q&A session."""
    from infer import load_model, get_inference

    display_banner()
    console.print("\n[bold green]Interactive Mode[/bold green]")
    console.print("Type your legal question, or 'exit' to quit.\n")

    try:
        load_model()
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(1)

    inference = get_inference()
    history = []

    while True:
        try:
            question = console.input("[bold cyan]Ask a legal question (or 'exit' to quit):[/bold cyan] ")
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

        # Display response
        console.print(Panel(
            result.response,
            title="[bold blue]LEGAL ORIENTATION[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        ))

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
    """Run the evaluation harness."""
    from pathlib import Path
    import json

    display_banner()
    console.print("\n[bold yellow]Evaluation Mode[/bold yellow]\n")

    eval_file = Path("eval/eval_set.json")
    if not eval_file.exists():
        console.print(f"[red]Eval set not found:[/red] {eval_file}")
        console.print("Run preprocess.py to generate the eval set.")
        raise typer.Exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    console.print(f"Loaded {len(eval_set)} evaluation questions\n")

    try:
        from infer import load_model, get_inference
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
    output_file = Path("eval/results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Evaluated {len(results)} questions[/green]")
    console.print(f"Results saved to {output_file}")


@app.command()
def info():
    """Display model metadata and training stats."""
    display_banner()

    table = Table(title="Model Information", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Base Model", "Mistral 7B Instruct v0.3")
    table.add_row("Fine-tuning", "LoRA (rank=16, alpha=32)")
    table.add_row("Training Framework", "Unsloth")
    table.add_row("Training Platform", "Kaggle (T4 GPU)")
    table.add_row("Domain", "Indian Law (BNS, BNSS, IPC, Consumer)")
    table.add_row("Dataset Size", "~2,800 instruction pairs")
    table.add_row("CLI Framework", "Typer + Rich")
    table.add_row("License", "MIT")

    console.print(table)


@app.command()
def version():
    """Show version and config."""
    console.print(f"THEMIS v{VERSION}")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Python: {__import__('sys').version.split()[0]}")


if __name__ == "__main__":
    app()
