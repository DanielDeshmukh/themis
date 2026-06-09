"""THEMIS CLI - Rich-powered terminal interface.

Commands:
    themis run                   - Show banner + run full pipeline
    themis ask "question"        - Single-shot legal Q&A
    themis chat                  - Interactive multi-turn session
    themis scrape                - Scrape legal data from India Code
    themis generate              - Generate synthetic Q&A pairs
    themis preprocess            - Merge and clean datasets
    themis eval                  - Run evaluation harness
    themis info                  - Model metadata
    themis version               - Version info
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

ASCII_ART = """
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
        "[bold white]The Parametric Legal Intelligence Engine for Indian Law[/bold white]"
    )
    console.print()


def display_info_panel():
    """Display the THEMIS info panel."""
    info_text = Text()
    info_text.append(f"  THEMIS", style="bold blue")
    info_text.append(" -- Indian Legal Intelligence Engine ", style="white")
    info_text.append(f"v{VERSION}", style="green")
    info_text.append("\n")
    info_text.append(f"  Model: {MODEL_NAME}", style="dim")
    info_text.append("\n")
    info_text.append("  Domain: BNS 2023 | BNSS 2023 | IPC | Consumer Law", style="dim")

    console.print(Panel(
        info_text,
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


# =============================================================================
# Main Commands
# =============================================================================

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """THEMIS - Indian Legal Intelligence Engine."""
    if ctx.invoked_subcommand is None:
        display_banner()
        display_info_panel()
        console.print("[dim]Type 'themis --help' to see available commands.[/dim]\n")


@app.command()
def run(
    skip_scrape: bool = typer.Option(False, "--skip-scrape", help="Skip scraping step"),
    no_api: bool = typer.Option(False, "--no-api", help="Use template generation only"),
):
    """Show banner and run the full data pipeline."""
    from data.scraper.indiacode import scrape_target_laws
    from data.synthetic.generate import generate_training_data
    from data.preprocess import preprocess_pipeline

    display_banner()
    display_info_panel()

    console.print("[bold yellow]Running Full Pipeline[/bold yellow]\n")

    # Step 1: Scrape
    if not skip_scrape:
        console.print("[cyan]Step 1/3: Scraping legal data...[/cyan]")
        scrape_target_laws()
        console.print("[green]Scraping complete![/green]\n")
    else:
        console.print("[dim]Step 1/3: Skipping scrape (--skip-scrape)[/dim]\n")

    # Step 2: Generate
    console.print("[cyan]Step 2/3: Generating Q&A pairs...[/cyan]")
    pairs = generate_training_data(use_api=not no_api)
    if pairs:
        console.print(f"[green]Generated {len(pairs)} pairs[/green]\n")
    else:
        console.print("[red]No pairs generated[/red]\n")

    # Step 3: Preprocess
    console.print("[cyan]Step 3/3: Preprocessing dataset...[/cyan]")
    preprocess_pipeline()

    console.print()
    console.print("[bold green]Pipeline complete![/bold green]")
    console.print("Next step: Upload training/finetune.py to Kaggle and run training.")
    console.print()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your legal question"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Ask a legal question and get a response."""
    from infer import load_model, get_inference

    display_banner()

    console.print(f"[bold cyan]Question:[/bold cyan] {question}\n")

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
    display_info_panel()

    console.print("[bold green]Interactive Mode[/bold green]")
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


# =============================================================================
# Data Pipeline Commands
# =============================================================================

@app.command()
def scrape(
    law: str = typer.Option(
        "all",
        "--law", "-l",
        help="Law to scrape: all, bns, bnss, bsa, cpa, rti, ipc"
    ),
    delay: float = typer.Option(1.0, "--delay", "-d", help="Delay between requests (seconds)"),
):
    """Scrape legal data from India Code and Indian Kanoon."""
    from data.scraper.indiacode import IndiaCodeScraper, scrape_target_laws

    display_banner()
    console.print("[bold yellow]Data Scraper[/bold yellow]\n")

    # Map short names to full law names
    LAW_MAP = {
        "bns": "Bharatiya Nyaya Sanhita, 2023",
        "bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "bsa": "Bharatiya Sakshya Adhiniyam, 2023",
        "cpa": "Consumer Protection Act, 2019",
        "rti": "Right to Information Act, 2005",
        "ipc": "Indian Penal Code, 1860",
        "pwa": "Payment of Wages Act, 1936",
        "ida": "Industrial Disputes Act, 1947",
    }

    if law.lower() == "all":
        console.print("[cyan]Scraping all target laws...[/cyan]\n")
        scrape_target_laws()
    elif law.lower() in LAW_MAP:
        full_name = LAW_MAP[law.lower()]
        console.print(f"[cyan]Scraping: {full_name}[/cyan]\n")
        scraper = IndiaCodeScraper(delay=delay)
        results = scraper.search_act(full_name)
        if results:
            best = results[0]
            act = scraper.scrape_act(best["handle_id"], full_name)
            scraper.save_act(act)
            console.print(f"\n[green]Scraped {len(act.sections)} sections from {act.name}[/green]")
        else:
            console.print(f"[red]No results found for '{full_name}'[/red]")
    else:
        console.print(f"[red]Unknown law: {law}[/red]")
        console.print(f"Available: {', '.join(LAW_MAP.keys())}, all")
        raise typer.Exit(1)

    console.print("\n[green]Scraping complete![/green]")


@app.command()
def generate(
    api: bool = typer.Option(
        True,
        "--api/--no-api",
        help="Use Groq API (free) or template fallback"
    ),
    max_pairs: int = typer.Option(
        None,
        "--max", "-m",
        help="Maximum Q&A pairs to generate"
    ),
):
    """Generate synthetic Q&A pairs from scraped data."""
    from data.synthetic.generate import generate_training_data

    display_banner()
    console.print("[bold yellow]Synthetic Data Generator[/bold yellow]\n")

    if api:
        import os
        if not os.environ.get("GROQ_API_KEY"):
            console.print("[yellow]GROQ_API_KEY not set. Using template fallback.[/yellow]")
            console.print("[dim]Get free key at: https://console.groq.com[/dim]\n")
            api = False

    pairs = generate_training_data(use_api=api)

    if pairs:
        console.print(f"\n[green]Generated {len(pairs)} Q&A pairs[/green]")
    else:
        console.print("[red]No pairs generated. Run 'themis scrape' first.[/red]")


@app.command()
def preprocess():
    """Merge, clean, and deduplicate all datasets."""
    from data.preprocess import preprocess_pipeline

    display_banner()
    console.print("[bold yellow]Data Preprocessing[/bold yellow]\n")

    preprocess_pipeline()

    console.print("\n[green]Preprocessing complete![/green]")


# =============================================================================
# Evaluation Command
# =============================================================================

@app.command()
def eval(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed results"),
):
    """Run the evaluation harness."""
    import json

    display_banner()
    console.print("[bold yellow]Evaluation Mode[/bold yellow]\n")

    eval_file = Path("eval/eval_set.json")
    if not eval_file.exists():
        console.print(f"[red]Eval set not found:[/red] {eval_file}")
        console.print("Run 'themis preprocess' to generate the eval set.")
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


# =============================================================================
# Info Commands
# =============================================================================

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
    console.print(f"Python: {sys.version.split()[0]}")


if __name__ == "__main__":
    app()
