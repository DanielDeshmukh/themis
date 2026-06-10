"""THEMIS CLI - Rich-powered terminal interface.

Commands:
    themis ask "question"   - Single-shot legal Q&A
    themis chat             - Interactive multi-turn session
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


if __name__ == "__main__":
    app()
