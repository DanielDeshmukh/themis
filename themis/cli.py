"""THEMIS CLI - Rich-powered terminal interface.

Commands:
    themis ask "question"   - Single-shot legal Q&A
    themis chat             - Interactive multi-turn session
    themis scrape           - Scrape legal data from India Code
    themis generate         - Generate synthetic Q&A pairs
    themis preprocess       - Merge and clean datasets
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
def scrape(
    law: str = typer.Option(
        "all",
        "--law", "-l",
        help="Law to scrape: all, bns, bnss, bsa, cpa, rti, ipc",
    ),
    delay: float = typer.Option(3.0, "--delay", "-d", help="Delay between requests (seconds)"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Show section text snippets"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-scrape even if already scraped"),
):
    """Scrape legal data from India Code."""
    from .data.scraper.indiacode import IndiaCodeScraper, scrape_target_laws

    display_banner()
    console.print("[bold yellow]Data Scraper[/bold yellow]\n")

    LAW_MAP = {
        "bns": "Bharatiya Nyaya Sanhita, 2023",
        "bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "bsa": "Bharatiya Sakshya Adhiniyam, 2023",
        "cpa": "Consumer Protection Act, 2019",
        "rti": "Right to Information Act, 2005",
        "ipc": "Indian Penal Code, 1860",
    }

    if law.lower() == "all":
        console.print("[cyan]Scraping all target laws...[/cyan]\n")
        scrape_target_laws(force=force)
    elif law.lower() in LAW_MAP:
        full_name = LAW_MAP[law.lower()]
        console.print(f"[cyan]Scraping: {full_name}[/cyan]\n")
        scraper = IndiaCodeScraper(delay=delay, verbose=verbose)

        # Check if already scraped
        from .data.scraper.indiacode import _is_already_scraped
        from .config import config
        if _is_already_scraped(full_name, config.raw_dir) and not force:
            console.print(f"[yellow]Already scraped: {full_name}[/yellow]")
            console.print("[yellow]Use --force to re-scrape[/yellow]")
            return

        results = scraper.search_act(full_name)
        if results:
            best = results[0]
            act = scraper.scrape_act(best["handle_id"], full_name)
            if act:
                scraper.save_act(act)
                console.print(f"\n[green]Scraped {len(act.sections)} sections from {act.name}[/green]")
            else:
                console.print(f"[red]Failed to fetch act page for {full_name}[/red]")
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
        help="Use Groq API (free) or template fallback",
    ),
    v2: bool = typer.Option(
        False,
        "--v2",
        help="Use v2 pipeline (3 Q&A per section + IPC mapping + abbreviations)",
    ),
    v3: bool = typer.Option(
        False,
        "--v3",
        help="Use v3 pipeline (10+ Q&A per section + scenarios + Groq API LLM generation)",
    ),
    max_pairs: int = typer.Option(
        None,
        "--max", "-m",
        help="Maximum Q&A pairs to generate",
    ),
    groq_limit: int = typer.Option(
        10000,
        "--groq-limit",
        help="Max LLM pairs from Groq API (v3 only)",
    ),
):
    """Generate synthetic Q&A pairs from scraped data."""
    display_banner()
    console.print("[bold yellow]Synthetic Data Generator[/bold yellow]\n")

    if v3:
        from .data.synthetic.generate_v3 import generate_v3_dataset
        pairs = generate_v3_dataset(
            pairs_per_section=10,
            use_groq=api,
            max_groq_pairs=groq_limit,
        )
    elif v2:
        from .data.synthetic.generate_v2 import generate_v2_dataset
        pairs = generate_v2_dataset(pairs_per_section=3)
    else:
        from .data.synthetic.generate import generate_training_data

        if api:
            import os
            if not os.environ.get("GROQ_API_KEY"):
                console.print("[yellow]GROQ_API_KEY not set. Using template fallback.[/yellow]")
                api = False

        pairs = generate_training_data(use_api=api)

    if pairs:
        console.print(f"\n[green]Generated {len(pairs)} Q&A pairs[/green]")
    else:
        console.print("[red]No pairs generated. Run 'themis scrape' first.[/red]")


@app.command()
def preprocess():
    """Merge, clean, and deduplicate all datasets."""
    from .data.preprocess import preprocess_pipeline

    display_banner()
    console.print("[bold yellow]Data Preprocessing[/bold yellow]\n")

    preprocess_pipeline()

    console.print("\n[green]Preprocessing complete![/green]")


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

    table.add_row("Status", "v1 trained | v2 in progress")
    table.add_row("Base Model", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    table.add_row("LoRA Adapter (v1)", "Daniel2503/themis-mistral-7b-lora")
    table.add_row("v1 LoRA Config", "rank=8, alpha=16, targets=[q_proj, v_proj]")
    table.add_row("v2 LoRA Config", "rank=16, alpha=32, targets=[q,k,v,o proj]")
    table.add_row("v1 Training Data", "1,939 Indian legal Q&A pairs")
    table.add_row("v2 Target", "10,000\u201315,000 pairs")
    table.add_row("Domain", "BNS 2023 | BNSS 2023 | IPC | Consumer Law | RTI Act")
    table.add_row("Quantization", "4-bit NF4 (bitsandbytes)")
    table.add_row("v1 Citation Accuracy", "~40% (hallucination rate ~60% on BNS queries)")
    table.add_row("v2 Target Accuracy", ">70% on criminal law queries")
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
