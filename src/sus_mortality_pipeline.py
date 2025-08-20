# %%
import os
import sys
import unicodedata
import polars as pl
import re
import requests
from datetime import datetime
from tqdm import tqdm
from pysus.online_data import SIM


# ||> Available SIM mortality data groups
sim_groups = {
    "CID9": (
        "CID-9 Classification",
        "Mortality data using International Classification of Diseases 9th revision",
    ),
    "CID10": (
        "CID-10 Classification",
        "Mortality data using International Classification of Diseases 10th revision",
    ),
}

# ||> Brazilian states
brazilian_states = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}

# ||> Brazilian regions with their states
brazilian_regions = {
    "NORTH": {
        "name": "North (Norte)",
        "states": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    },
    "NORTHEAST": {
        "name": "Northeast (Nordeste)",
        "states": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    },
    "SOUTHEAST": {"name": "Southeast (Sudeste)", "states": ["ES", "MG", "RJ", "SP"]},
    "SOUTH": {"name": "South (Sul)", "states": ["PR", "RS", "SC"]},
    "CENTER_WEST": {"name": "Center-West (Centro-Oeste)", "states": ["GO", "MT", "MS"]},
}


# ||> SHARED UTILITY FUNCTIONS
def get_visual_width(text):
    """Calculate the visual width of a string, accounting for emojis and wide characters."""
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ("F", "W"):
            width += 2  # |> Full-width or wide characters
        elif unicodedata.category(char).startswith("So"):
            width += 2  # |> Symbol characters (including most emojis)
        else:
            width += 1  # |> Normal characters
    return width


def ensure_directory_exists(dir_path):
    """Ensures that a directory exists, creating it if necessary."""
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            print(f"\nError: Path '{dir_path}' exists but is not a directory.")
            sys.stdout.flush()
            return False
        return True
    else:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Successfully created directory: {dir_path}")
            sys.stdout.flush()
            return True
        except OSError as e:
            print(f"❌ Error creating directory {dir_path}: {e}")
            sys.stdout.flush()
            return False
        except Exception as e:
            print(
                f"❌ An unexpected error occurred while creating directory {dir_path}: {e}"
            )
            sys.stdout.flush()
            return False


def get_brazilian_state_codes():
    """Get a set of valid Brazilian state codes."""
    return set(brazilian_states.keys())


def is_state_directory(dir_path):
    """Check if a directory path represents a Brazilian state directory."""
    if not dir_path:
        return False
    dir_name = os.path.basename(os.path.normpath(dir_path)).upper()
    return dir_name in get_brazilian_state_codes()


def get_state_aware_directory(base_dir, state_code, subfolder=None):
    """
    Get state-aware directory ensuring proper organization.

    Args:
        base_dir (str): Base directory
        state_code (str): State code (e.g., 'SP')
        subfolder (str): Optional subfolder ('extracted' or 'processed')

    Returns:
        str: Properly organized directory path
    """
    # ||> If user specified a state directory, go up one level
    if is_state_directory(base_dir):
        base_dir = os.path.dirname(base_dir)

    # ||> Build path: base_dir/state/subfolder (if subfolder specified)
    if subfolder:
        final_dir = os.path.join(base_dir, state_code, subfolder)
    else:
        final_dir = os.path.join(base_dir, state_code)

    return os.path.abspath(final_dir)


# ||> EXTRACTION FUNCTIONS
def determine_groups_from_years(years):
    """Automatically determine which SIM groups to use based on years."""
    cid9_years = [year for year in years if year <= 1995]
    cid10_years = [year for year in years if year >= 1996]

    groups = []
    if cid9_years:
        groups.append("CID9")
    if cid10_years:
        groups.append("CID10")

    return groups


def get_user_state_selection():
    """Prompt user to select Brazilian states or regions for data download."""
    title = "🌍 Brazilian States and Regions:"
    print("\n" + "=" * get_visual_width(title))
    print(f"{title}")
    print("=" * get_visual_width(title))

    print("You can select:")
    print("  1. Individual states")
    print("  2. Entire regions")
    print("  3. Mix of states and regions")
    print("  4. All states (type 'all')")
    print()

    # ||> Display regions
    print("📍 Available Regions:")
    for code, info in brazilian_regions.items():
        states_str = ", ".join(info["states"])
        print(f" {code}: {info['name']}")
        print(f" States: {states_str}")
        print()

    # ||> Display states in columns
    print("📍 Individual States:")
    state_items = list(brazilian_states.items())
    for i in range(0, len(state_items), 3):
        line = ""
        for j in range(3):
            if i + j < len(state_items):
                code, name = state_items[i + j]
                line += f"{code}: {name:<20} "
        print(f"  {line}")
    print()

    while True:
        user_input = input(
            "Enter state codes, region codes, or 'all' for all states (e.g., 'SP,RJ', 'SOUTHEAST', 'SP,NORTH', 'all'): "
        ).strip()

        if user_input.lower() == "all":
            selected_states = list(brazilian_states.keys())
            print(f"✅ Selected ALL states ({len(selected_states)} states)")
            break

        try:
            selected_codes = [code.strip().upper() for code in user_input.split(",")]
            selected_states = set()
            invalid_codes = []

            for code in selected_codes:
                if code in brazilian_states:
                    selected_states.add(code)
                elif code in brazilian_regions:
                    selected_states.update(brazilian_regions[code]["states"])
                else:
                    invalid_codes.append(code)

            if invalid_codes:
                print(f"Error: Invalid codes: {invalid_codes}")
                print(f"Valid state codes: {list(brazilian_states.keys())}")
                print(f"Valid region codes: {list(brazilian_regions.keys())}")
                print("Or type 'all' to select all states")
                continue

            if not selected_states:
                print(
                    "Please select at least one state or region, or type 'all' for all states."
                )
                continue

            selected_states = sorted(list(selected_states))
            print(f"\n✅ Selected states ({len(selected_states)}):")

            if len(selected_states) == len(brazilian_states):
                print("   • All Brazilian states selected!")
            else:
                for code in selected_states:
                    print(f"   • {code}: {brazilian_states[code]}")
            break

        except Exception as e:
            print(f"Error parsing input: {e}")
            print("Please enter state codes, region codes, or 'all' for all states.")

    return selected_states


def get_user_year_selection():
    """Prompt user to select years for data download."""
    current_year = datetime.now().year
    max_year = current_year - 2
    min_year = 1979

    title = "📅 Year Selection:"
    print("\n\n" + "=" * get_visual_width(title))
    print(f"{title}")
    print("=" * get_visual_width(title))
    print("You can enter:")
    print("  - Individual years: '2020,2021,2022'")
    print("  - Year ranges: '2018-2022'")
    print("  - Mixed: '2015,2018-2020,2022'")
    print(f"  - Valid range: {min_year}-{max_year}")
    print()

    while True:
        user_input = input("Enter years or ranges: ").strip()

        try:
            selected_years = []
            parts = [part.strip() for part in user_input.split(",")]

            for part in parts:
                if "-" in part:
                    start_str, end_str = part.split("-", 1)
                    start_year = int(start_str.strip())
                    end_year = int(end_str.strip())

                    if start_year > end_year:
                        print(
                            f"Error: Invalid range {part}. Start year must be <= end year."
                        )
                        selected_years = []
                        break

                    selected_years.extend(range(start_year, end_year + 1))
                else:
                    year = int(part)
                    selected_years.append(year)

            if not selected_years:
                continue

            selected_years = sorted(list(set(selected_years)))

            invalid_years = [
                year for year in selected_years if year < min_year or year > max_year
            ]
            if invalid_years:
                print(
                    f"Error: Invalid years {invalid_years}. Years should be between {min_year} and {max_year}."
                )
                continue

            break

        except ValueError as e:
            print(f"Error parsing years: {e}")
            print("Please enter valid years or ranges.")

    print(f"✅ Selected years ({len(selected_years)}): {selected_years}")
    return selected_years


def check_existing_files(groups, states, years, output_dir):
    """Check for existing files and get user preference for handling them."""
    existing_files = set()

    for group in groups:
        for state in states:
            for year in years:
                extracted_dir = get_state_aware_directory(
                    output_dir, state, "extracted"
                )
                if os.path.exists(extracted_dir):
                    year_pattern = f"{state}{year}"
                    for item in os.listdir(extracted_dir):
                        if (
                            item.endswith(".parquet") or "parquet" in item.lower()
                        ) and year_pattern in item:
                            existing_files.add(os.path.join(extracted_dir, item))

    if not existing_files:
        return "proceed"

    existing_files = sorted(list(existing_files))
    print(
        f"\n🚨  Found {len(existing_files)} existing file(s) that match your request:"
    )
    for file_path in existing_files[:5]:
        print(f"   {file_path}")
    if len(existing_files) > 5:
        print(f"   ... and {len(existing_files) - 5} more files")

    print("\nHow would you like to handle these existing files?")
    print("  1. Skip downloads for existing files (download only missing ones)")
    print("  2. Overwrite existing files")
    print("  3. Cancel operation")

    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == "1":
            return "skip"
        elif choice == "2":
            return "overwrite"
        elif choice == "3":
            return "cancel"
        else:
            print("Please enter 1, 2, or 3.")


def download_sim_data(groups, states, years, output_dir, file_handling="overwrite"):
    """Download SIM mortality data for specified parameters."""
    downloaded_data = {}
    total_downloads = len(groups) * len(states) * len(years)
    current_download = 0
    skipped_downloads = 0
    successful_downloads = 0

    for group in groups:
        if group not in downloaded_data:
            downloaded_data[group] = {}

        for state in states:
            if state not in downloaded_data[group]:
                downloaded_data[group][state] = {}

            for year in years:
                current_download += 1

                try:
                    # ||> Create extracted directory for this state
                    extracted_dir = get_state_aware_directory(
                        output_dir, state, "extracted"
                    )

                    print(
                        f"[{successful_downloads + skipped_downloads + 1}/{total_downloads}] Downloading {group}-{state}-{year}..."
                    )

                    if file_handling == "skip" and os.path.exists(extracted_dir):
                        year_pattern = f"{state}{year}"
                        existing_files = [
                            item
                            for item in os.listdir(extracted_dir)
                            if (item.endswith(".parquet") or "parquet" in item.lower())
                            and year_pattern in item
                        ]
                        if existing_files:
                            print(f"⏭️  Skipping {group}-{state}-{year} (file exists)")
                            downloaded_data[group][state][year] = "skipped"
                            skipped_downloads += 1
                            continue

                    sys.stdout.flush()

                    if not ensure_directory_exists(extracted_dir):
                        print(f"❌ Failed to create directory: {extracted_dir}")
                        downloaded_data[group][state][year] = None
                        skipped_downloads += 1
                        continue

                    # ||> Download data using pysus
                    downloaded_result = SIM.download(
                        groups=[group],
                        states=[state],
                        years=[year],
                        data_dir=extracted_dir,
                    )

                    if downloaded_result:
                        downloaded_data[group][state][year] = downloaded_result
                        successful_downloads += 1
                        print(f"✅ Downloaded {group}-{state}-{year}")
                    else:
                        print(f"❌ Failed to download {group}-{state}-{year}")
                        downloaded_data[group][state][year] = None
                        skipped_downloads += 1

                except Exception as e:
                    print(f"❌ Error downloading {group}-{state}-{year}: {str(e)}")
                    downloaded_data[group][state][year] = None
                    skipped_downloads += 1

                print()

    return downloaded_data, successful_downloads


def cleanup_pysus_cache():
    """Remove the pysus cache directory from the user's home directory."""
    home_dir = os.path.expanduser("~")
    pysus_cache_dir = os.path.join(home_dir, "pysus")

    try:
        if os.path.exists(pysus_cache_dir):
            import shutil

            shutil.rmtree(pysus_cache_dir)
            print("✅ Cleaned up pysus cache directory")
            return True
        else:
            return True
    except Exception:
        print(
            f"📢  Error cleaning up pysus cache. You may manually delete: {pysus_cache_dir}"
        )
        return False


# ||> PROCESSING FUNCTIONS
def parse_date_string(date_str):
    """Parse date string from DDMMYYYY format to datetime."""
    try:
        if (
            date_str is None
            or str(date_str).strip() == ""
            or str(date_str).strip() == "null"
        ):
            return None

        date_str = str(date_str).strip()
        if len(date_str) != 8:
            return None

        day = int(date_str[:2])
        month = int(date_str[2:4])
        year = int(date_str[4:8])

        return datetime(year, month, day)
    except (ValueError, TypeError):
        return None


def calculate_age_at_death(birth_date, death_date):
    """Calculate age at death in years with month-based rounding."""
    try:
        if birth_date is None or death_date is None:
            return None

        age_years = death_date.year - birth_date.year
        months_diff = death_date.month - birth_date.month

        if death_date.day < birth_date.day:
            months_diff -= 1

        if months_diff < 0:
            age_years -= 1
            months_diff += 12

        if months_diff >= 6:
            age_years += 1

        return age_years if age_years >= 0 else None
    except Exception:
        return None


def get_age_range_label(age, age_ranges):
    """Get the age range label for a given age based on defined ranges."""
    if age is None:
        return None

    for start, end in age_ranges:
        if start <= age <= end:
            return f"age_{start}_{end}"
    return None


def parse_age_ranges(age_ranges_str):
    """Parse age ranges string into list of tuples for processing."""
    try:
        ranges = []
        for range_str in age_ranges_str.split(","):
            start, end = map(int, range_str.strip().split("-"))
            ranges.append((start, end))
        return ranges
    except ValueError:
        print("Error: Invalid age range format. Use format like '15-30,31-45,46-55'")
        return None


def extract_state_from_path(file_path):
    """Extract state abbreviation from SUS file/folder path."""
    filename = os.path.basename(file_path)
    match = re.search(r"DO([A-Z]{2})\d{4}", filename)
    return match.group(1) if match else None


def fetch_ibge_municipalities():
    """Fetch municipality data from IBGE API and convert to Polars DataFrame."""
    try:
        print("🌐 Fetching IBGE municipality data...")
        response = requests.get(
            "https://servicodados.ibge.gov.br/api/v1/localidades/municipios", timeout=30
        )
        response.raise_for_status()

        municipalities_data = response.json()
        
        if not municipalities_data:
            print("❌ No municipality data received from IBGE API")
            return None

        # ||> Convert to format expected by other functions
        data = []
        skipped_items = 0
        
        for item in municipalities_data:
            try:
                # ||> Check if all required nested fields exist
                if (item and 
                    "id" in item and 
                    "microrregiao" in item and 
                    item["microrregiao"] and
                    "mesorregiao" in item["microrregiao"] and
                    item["microrregiao"]["mesorregiao"] and
                    "UF" in item["microrregiao"]["mesorregiao"] and
                    item["microrregiao"]["mesorregiao"]["UF"] and
                    "sigla" in item["microrregiao"]["mesorregiao"]["UF"]):
                    
                    data.append({
                        "municipio-id": str(item["id"]),
                        "UF-sigla": item["microrregiao"]["mesorregiao"]["UF"]["sigla"],
                    })
                else:
                    skipped_items += 1
                    
            except (KeyError, TypeError):
                skipped_items += 1
                continue

        if not data:
            print("❌ No valid municipality data found in IBGE response")
            return None

        df = pl.DataFrame(data)
        print(f"✅ Fetched {len(df)} municipalities from IBGE")
        if skipped_items > 0:
            print(f"📢 Skipped {skipped_items} items with incomplete data")
        return df

    except Exception as e:
        print(f"❌ Error fetching IBGE data: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_cid_codes(cid_list, available_cids):
    """Validate user-provided CID codes against available codes in data."""
    # ||> Create mappings to preserve original format from data
    available_cids_clean = [str(cid).strip() if cid else "" for cid in available_cids]
    available_cids_upper = [cid.upper() for cid in available_cids_clean if cid]
    
    # ||> Create mapping from uppercase to original format
    upper_to_original = {cid.upper(): cid for cid in available_cids_clean if cid}
    
    cid_list_upper = [str(cid).upper().strip() for cid in cid_list if cid]

    valid_cids = []
    invalid_cids = []

    for user_input in cid_list_upper:
        found_matches = []
        
        # ||> Search for all matches (both exact and partial)
        for available_cid_upper in available_cids_upper:
            if available_cid_upper.startswith(user_input):
                original_format = upper_to_original[available_cid_upper]
                if original_format not in found_matches:
                    found_matches.append(original_format)
        
        if found_matches:
            valid_cids.extend(found_matches)
            # ||> Only show summary for large matches, stay quiet for exact matches
            if len(found_matches) > 10:
                print(f"✅ '{user_input}' matched {len(found_matches)} CID(s)")
        else:
            invalid_cids.append(user_input)
            print(f"❌ '{user_input}' - no matches found")

    # ||> Remove duplicates while preserving order
    unique_valid_cids = list(dict.fromkeys(valid_cids))

    return unique_valid_cids, invalid_cids


def get_user_cid_selection():
    """Get CID code selection from user with validation."""
    print("Enter the CID codes you want to analyze.")
    print("Format: One letter + digits (e.g., A01, B02.1, C78, C34.1)")
    print("")
    print("📋 Matching Options:")
    print("  • Exact match: 'C341' matches only C341")
    print("  • Partial match: 'C34' matches C341, C342, C349, etc.")
    print("  • Category match: 'C3' matches all CIDs starting with C3")
    print("  • Letter match: 'C' matches all CIDs starting with C")
    print("")
    print(
        "You can enter multiple codes separated by commas (e.g., 'C3', 'A', 'C341,B02', 'I00,J')"
    )
    sys.stdout.flush()

    while True:
        user_input = input("\nEnter CID codes: ").strip()

        if not user_input:
            print("Please enter at least one CID code.")
            continue

        try:
            cid_codes = [code.strip().upper() for code in user_input.split(",")]
            # ||> Basic format validation - allow single letters or letter+digits
            valid_format = True
            for code in cid_codes:
                # ||> Allow single letter (A, B, C) or letter followed by digits/dots (A01, B02.1, C34)
                if not re.match(r"^[A-Z]([0-9].*)?$", code):
                    print(f"Invalid format for CID code: {code}")
                    print("Format should be: single letter (A, C, J) or letter + digits (A01, C34.1)")
                    valid_format = False

            if not valid_format:
                continue

            print(f"✅ Selected CID codes: {cid_codes}")
            return cid_codes

        except Exception as e:
            print(f"Error parsing CID codes: {e}")

    return cid_codes


def complete_municipality_codes(df, ibge_df, state_code):
    """Complete 6-digit municipality codes to 7-digit using IBGE data."""
    state_municipalities = ibge_df.filter(pl.col("UF-sigla") == state_code)

    ibge_codes = state_municipalities.select(
        [pl.col("municipio-id").cast(pl.String).alias("full_code")]
    ).with_columns([pl.col("full_code").str.slice(0, 6).alias("prefix_6")])

    code_mapping = {
        row["prefix_6"]: row["full_code"]
        for row in ibge_codes.to_dicts()
        if len(row["full_code"]) == 7
    }

    df = df.with_columns(
        [
            pl.when(pl.col("CODMUNRES").str.len_chars() == 6)
            .then(
                pl.col("CODMUNRES").map_elements(
                    lambda x: code_mapping.get(x, None), return_dtype=pl.String
                )
            )
            .otherwise(pl.col("CODMUNRES"))
            .alias("CODMUNRES")
        ]
    ).filter(pl.col("CODMUNRES").is_not_null())

    return df


def create_complete_grid(
    df,
    ibge_df,
    state_code,
    selected_cids,
    age_ranges=None,
    use_gender=False,
    aggregate_cids=False,
):
    """
    Create complete grid ensuring all municipalities have data for all CIDs and months.

    This function ensures that every municipality (including new IBGE municipalities)
    has entries for every selected CID and every month, filling missing combinations
    with zero counts.

    Args:
        df (pl.DataFrame): Processed mortality data.
        ibge_df (pl.DataFrame): IBGE municipality data.
        state_code (str): State abbreviation.
        selected_cids (list): List of selected CID codes.
        age_ranges (list, optional): Age ranges for analysis.
        use_gender (bool, optional): Whether to include gender analysis.
        aggregate_cids (bool, optional): Whether to aggregate all CIDs together.

    Returns:
        pl.DataFrame: Complete grid with all municipality-CID-month combinations.
    """

    # ||> Get all municipalities for the state from IBGE
    state_municipalities = ibge_df.filter(pl.col("UF-sigla") == state_code)
    all_municipality_codes = state_municipalities.select(
        [pl.col("municipio-id").cast(pl.String).alias("residence_municipality_code")]
    )

    # ||> Generate complete set of months for the year(s) in the data
    if len(df) > 0:
        # ||> Get the year range from the data
        min_date = df.select("time").min().item()
        max_date = df.select("time").max().item()

        # ||> Generate all months from min year to max year
        import datetime

        start_year = min_date.year
        end_year = max_date.year

        all_month_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                month_date = datetime.date(year, month, 1)
                all_month_dates.append(month_date)

        # ||> Create DataFrame with all months
        all_months = pl.DataFrame({"time": all_month_dates}).sort("time")
    else:
        # ||> If no data, create a basic month range (this shouldn't happen normally)
        return pl.DataFrame()

    # ||> Create complete grid based on aggregation mode
    if aggregate_cids:
        # ||> For aggregated CIDs, only create time-municipality grid
        grid_rows = []
        for month_row in all_months.iter_rows(named=True):
            for munic_row in all_municipality_codes.iter_rows(named=True):
                grid_rows.append(
                    {
                        "time": month_row["time"],
                        "residence_municipality_code": munic_row[
                            "residence_municipality_code"
                        ],
                    }
                )

        # ||> Create grid DataFrame for aggregated mode
        grid_df = pl.DataFrame(
            grid_rows,
            schema={"time": pl.Date, "residence_municipality_code": pl.String},
        )
        join_cols = ["time", "residence_municipality_code"]
    else:
        # ||> For separate CIDs, create time-municipality-CID grid
        grid_rows = []
        for month_row in all_months.iter_rows(named=True):
            for munic_row in all_municipality_codes.iter_rows(named=True):
                for cid in selected_cids:
                    grid_rows.append(
                        {
                            "time": month_row["time"],
                            "residence_municipality_code": munic_row[
                                "residence_municipality_code"
                            ],
                            "CID_value": cid,
                        }
                    )

        # ||> Create grid DataFrame for separate CID mode
        grid_df = pl.DataFrame(
            grid_rows,
            schema={
                "time": pl.Date,
                "residence_municipality_code": pl.String,
                "CID_value": pl.String,
            },
        )
        join_cols = ["time", "residence_municipality_code", "CID_value"]

    # ||> Prepare aggregation columns
    agg_cols = ["mort_total_count"]

    if use_gender:
        agg_cols.extend(["mort_male_count", "mort_female_count"])

        if age_ranges:
            for start, end in age_ranges:
                age_label = f"age_{start}_{end}"
                # ||> Add total age range counts (all genders)
                agg_cols.append(f"mort_{age_label}")
                # ||> Add gender-specific age counts
                agg_cols.extend([f"mort_male_{age_label}", f"mort_female_{age_label}"])
    elif age_ranges:
        for start, end in age_ranges:
            age_label = f"age_{start}_{end}"
            agg_cols.append(f"mort_{age_label}")

    # ||> Join with actual data and fill missing values with 0
    if len(df) > 0:
        # ||> Try a direct join approach with proper column selection
        try:
            # ||> Select only the columns needed for joining and aggregation
            data_for_join = df.select(
                join_cols + [col for col in df.columns if col not in join_cols]
            )

            result_df = grid_df.join(data_for_join, on=join_cols, how="left")

        except Exception as e:
            print(f"Join failed: {e}")
            print("Using fallback approach...")
            sys.stdout.flush()

            # ||> Fallback: Add missing columns to grid and concatenate
            missing_cols = [col for col in df.columns if col not in grid_df.columns]
            for col in missing_cols:
                grid_df = grid_df.with_columns(pl.lit(None).alias(col))

            # ||> Use concat and then group by to aggregate
            result_df = pl.concat([grid_df, df], how="diagonal")
            result_df = result_df.group_by(join_cols).agg(
                [
                    pl.col(col).sum().alias(col)
                    for col in missing_cols
                    if col in agg_cols
                ]
            )

        # ||> Fill null values with 0 for all count columns
        existing_agg_cols = [col for col in agg_cols if col in result_df.columns]
        missing_agg_cols = [col for col in agg_cols if col not in result_df.columns]

        # ||> Fill nulls for existing columns
        if existing_agg_cols:
            fill_exprs = []
            for col in existing_agg_cols:
                fill_exprs.append(pl.col(col).fill_null(0).alias(col))
            result_df = result_df.with_columns(fill_exprs)

        # ||> Add missing columns with zero values
        for col in missing_agg_cols:
            result_df = result_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias(col))

    else:
        # ||> If no data, create grid with all zeros
        for col in agg_cols:
            grid_df = grid_df.with_columns(pl.lit(0).alias(col))
        result_df = grid_df

    return result_df


def process_parquet_file(
    path,
    age_ranges=None,
    use_gender=False,
    selected_cids=None,
    ibge_df=None,
    aggregate_cids=False,
):
    """Process a single parquet file or directory and return aggregated mortality data."""

    # ||> Determine if we should remove CID_value column (aggregated mode or single CID)
    single_cid_mode = selected_cids and len(selected_cids) == 1 and not aggregate_cids
    remove_cid_column = aggregate_cids or single_cid_mode

    # ||> Required columns for processing
    required_columns = [
        "TIPOBITO",
        "DTOBITO",
        "DTNASC",
        "SEXO",
        "CODMUNRES",
        "CAUSABAS",
    ]

    try:
        # ||> Read parquet data lazily
        lf = pl.scan_parquet(path)

        # ||> Check if required columns exist
        available_columns = lf.collect_schema().names()
        missing_columns = [
            col for col in required_columns if col not in available_columns
        ]
        if missing_columns:
            print(f"❌ Error: Missing required columns: {missing_columns}")
            sys.stdout.flush()
            return None

        # ||> Process data lazily - strip spaces, convert types, filter
        lf = (
            lf.select(required_columns)
            .with_columns(
                [
                    pl.col("TIPOBITO").str.strip_chars(),
                    pl.col("DTOBITO").str.strip_chars(),
                    pl.col("DTNASC").str.strip_chars(),
                    pl.col("SEXO").str.strip_chars(),
                    pl.col("CODMUNRES").str.strip_chars(),
                    pl.col("CAUSABAS").str.strip_chars(),
                ]
            )
            .with_columns(
                [
                    pl.col("TIPOBITO").cast(pl.Int32, strict=False),
                    pl.col("SEXO").cast(pl.Int32, strict=False),
                ]
            )
            .filter(pl.col("TIPOBITO") == 2)  # |> Filter for TIPOBITO == 2
            .filter(
                pl.col("SEXO").is_not_null()
                & pl.col("CODMUNRES").is_not_null()
                & pl.col("CAUSABAS").is_not_null()
                & pl.col("DTOBITO").is_not_null()
                & pl.col("DTNASC").is_not_null()
            )
        )

        # ||> Collect for date parsing
        df = lf.collect()

        if len(df) == 0:
            print("❌ No valid records remaining after filtering")
            sys.stdout.flush()
            return None

        # ||> Parse dates and calculate ages
        with tqdm(
            total=3, desc="Processing dates", unit="step", leave=False, ncols=80
        ) as pbar:
            pbar.set_description("Parsing death dates")
            df = df.with_columns(
                [
                    pl.col("DTOBITO")
                    .map_elements(parse_date_string, return_dtype=pl.Datetime)
                    .alias("death_date")
                ]
            )
            pbar.update(1)

            pbar.set_description("Parsing birth dates")
            df = df.with_columns(
                [
                    pl.col("DTNASC")
                    .map_elements(parse_date_string, return_dtype=pl.Datetime)
                    .alias("birth_date")
                ]
            )
            pbar.update(1)

            pbar.set_description("Calculating ages")
            df = df.with_columns(
                [
                    pl.struct(["birth_date", "death_date"])
                    .map_elements(
                        lambda x: calculate_age_at_death(
                            x["birth_date"], x["death_date"]
                        )
                        if x["birth_date"] and x["death_date"]
                        else None,
                        return_dtype=pl.Int32,
                    )
                    .alias("age_at_death")
                ]
            )
            pbar.update(1)

        # ||> Filter out rows with invalid dates or ages
        df = df.filter(
            pl.col("death_date").is_not_null()
            & pl.col("birth_date").is_not_null()
            & pl.col("age_at_death").is_not_null()
        )

        if len(df) == 0:
            print("❌ No valid records remaining after date processing")
            sys.stdout.flush()
            return None

        # ||> Create month column (first day of month) as date only
        df = df.with_columns(
            [pl.col("death_date").dt.truncate("1mo").dt.date().alias("time")]
        )

        # ||> Filter out rows with invalid city codes (ending with 0000)
        df = df.filter(~pl.col("CODMUNRES").str.ends_with("0000"))

        if len(df) == 0:
            print("❌ No valid records remaining after city code filtering")
            sys.stdout.flush()
            return None

        # ||> Filter by selected CID codes if provided
        if selected_cids:
            available_cids = df.select("CAUSABAS").unique().to_series().to_list()
            available_cids = [
                cid.strip() if isinstance(cid, str) else cid for cid in available_cids
            ]
            valid_cids, invalid_cids = validate_cid_codes(selected_cids, available_cids)

            if not valid_cids:
                print(
                    f"❌ None of the selected CID codes {selected_cids} found in this file"
                )
                sys.stdout.flush()
                return None

            if invalid_cids:
                print(f"🚨  CID codes not found in file: {invalid_cids}")
                sys.stdout.flush()

            # ||> Filter data for selected CIDs
            df = df.with_columns(
                [pl.col("CAUSABAS").str.strip_chars().alias("CAUSABAS")]
            ).filter(pl.col("CAUSABAS").is_in(valid_cids))

            if len(df) == 0:
                print("❌ No records remaining after CID filtering")
                sys.stdout.flush()
                return None

        # ||> Complete municipality codes using IBGE data if provided
        if ibge_df is not None:
            state_code = extract_state_from_path(path)
            if state_code:
                df = complete_municipality_codes(df, ibge_df, state_code)

                if len(df) == 0:
                    print("❌ No records remaining after municipality code validation")
                    sys.stdout.flush()
                    return None

        # ||> Add age range column if needed
        if age_ranges:
            df = df.with_columns(
                [
                    pl.col("age_at_death")
                    .map_elements(
                        lambda age: get_age_range_label(age, age_ranges),
                        return_dtype=pl.String,
                    )
                    .alias("age_range")
                ]
            )

        # ||> Perform aggregation using Polars group_by
        if remove_cid_column:
            group_cols = ["time", "CODMUNRES"]
        else:
            group_cols = ["time", "CODMUNRES", "CAUSABAS"]

        # ||> Create base aggregation
        base_agg = df.group_by(group_cols).agg([pl.len().alias("mort_total_count")])
        result_df = base_agg

        # ||> Add gender-specific counts if requested
        if use_gender:
            with tqdm(
                total=2, desc="Gender aggregations", unit="step", leave=False, ncols=80
            ) as pbar:
                pbar.set_description("Processing male counts")
                male_agg = (
                    df.filter(pl.col("SEXO") == 1)
                    .group_by(group_cols)
                    .agg([pl.len().alias("mort_male_count")])
                )
                pbar.update(1)

                pbar.set_description("Processing female counts")
                female_agg = (
                    df.filter(pl.col("SEXO") == 2)
                    .group_by(group_cols)
                    .agg([pl.len().alias("mort_female_count")])
                )
                pbar.update(1)

            # ||> Join gender counts
            result_df = result_df.join(male_agg, on=group_cols, how="left").fill_null(0)
            result_df = result_df.join(female_agg, on=group_cols, how="left").fill_null(
                0
            )

            # ||> Add age range counts by gender if age ranges are specified
            if age_ranges:
                with tqdm(
                    age_ranges,
                    desc="Age-gender aggregations",
                    unit="range",
                    leave=False,
                    ncols=80,
                ) as age_pbar:
                    for start, end in age_pbar:
                        age_label = f"age_{start}_{end}"
                        age_pbar.set_description(f"Processing {age_label}")

                        # ||> Total age range counts
                        age_agg = (
                            df.filter(pl.col("age_range") == age_label)
                            .group_by(group_cols)
                            .agg([pl.len().alias(f"mort_{age_label}")])
                        )

                        # ||> Male age range counts
                        male_age_agg = (
                            df.filter(
                                (pl.col("SEXO") == 1)
                                & (pl.col("age_range") == age_label)
                            )
                            .group_by(group_cols)
                            .agg([pl.len().alias(f"mort_male_{age_label}")])
                        )

                        # ||> Female age range counts
                        female_age_agg = (
                            df.filter(
                                (pl.col("SEXO") == 2)
                                & (pl.col("age_range") == age_label)
                            )
                            .group_by(group_cols)
                            .agg([pl.len().alias(f"mort_female_{age_label}")])
                        )

                        # ||> Join all age range counts
                        result_df = result_df.join(
                            age_agg, on=group_cols, how="left"
                        ).fill_null(0)
                        result_df = result_df.join(
                            male_age_agg, on=group_cols, how="left"
                        ).fill_null(0)
                        result_df = result_df.join(
                            female_age_agg, on=group_cols, how="left"
                        ).fill_null(0)

        # ||> Add age range counts (total) if age ranges are specified but gender is not
        elif age_ranges:
            with tqdm(
                age_ranges,
                desc="Age range aggregations",
                unit="range",
                leave=False,
                ncols=80,
            ) as age_pbar:
                for start, end in age_pbar:
                    age_label = f"age_{start}_{end}"
                    age_pbar.set_description(f"Processing {age_label}")

                    age_agg = (
                        df.filter(pl.col("age_range") == age_label)
                        .group_by(group_cols)
                        .agg([pl.len().alias(f"mort_{age_label}")])
                    )

                    result_df = result_df.join(
                        age_agg, on=group_cols, how="left"
                    ).fill_null(0)

        # ||> Rename columns to match expected output format
        if remove_cid_column:
            result_df = result_df.rename({"CODMUNRES": "residence_municipality_code"})
        else:
            result_df = result_df.rename(
                {"CODMUNRES": "residence_municipality_code", "CAUSABAS": "CID_value"}
            )

        # ||> Create complete grid if IBGE data is available
        if ibge_df is not None and selected_cids:
            state_code = extract_state_from_path(path)
            if state_code:
                result_df = create_complete_grid(
                    result_df,
                    ibge_df,
                    state_code,
                    selected_cids,
                    age_ranges=age_ranges,
                    use_gender=use_gender,
                    aggregate_cids=aggregate_cids,
                )
            else:
                print("🚨  Warning: Could not determine state code from path, skipping grid completion")
                sys.stdout.flush()

        # ||> Sort by appropriate columns
        if remove_cid_column:
            result_df = result_df.sort(["time", "residence_municipality_code"])
        else:
            result_df = result_df.sort(
                ["time", "residence_municipality_code", "CID_value"]
            )

        return result_df

    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        sys.stdout.flush()
        return None


def get_downloaded_files(downloaded_data, output_dir):
    """Get list of successfully downloaded files for processing."""
    downloaded_files = []

    for group, states_data in downloaded_data.items():
        for state, years_data in states_data.items():
            for year, result in years_data.items():
                if result and result != "skipped":
                    # ||> Look for parquet files in the extracted directory
                    extracted_dir = get_state_aware_directory(
                        output_dir, state, "extracted"
                    )
                    if os.path.exists(extracted_dir):
                        for item in os.listdir(extracted_dir):
                            if item.endswith(".parquet") or "parquet" in item.lower():
                                if f"{state}{year}" in item:
                                    file_path = os.path.join(extracted_dir, item)
                                    downloaded_files.append(file_path)

    return downloaded_files


# ||> MAIN PIPELINE FUNCTION
def main():
    """Main function that orchestrates the complete SUS mortality data pipeline."""

    # ||> Get script and parent directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir_of_script_folder = os.path.dirname(script_dir)

    title = "🔄 SUS Mortality Data Pipeline"
    print("")
    print("=" * get_visual_width(title))
    print(title)
    print("=" * get_visual_width(title))
    print("This pipeline downloads SUS mortality data and processes it into")
    print("aggregated CSV files with municipality coordinates.")
    print("")
    print("📁 File Organization:")
    print("   • Downloaded files: [output_dir]/[state]/extracted/")
    print("   • Processed files: [output_dir]/[state]/processed/")
    print("")
    print("📢  Note: CID classification is automatically selected based on years:")
    print("   • Years ≤ 1995: CID-9 classification")
    print("   • Years ≥ 1996: CID-10 classification")
    print("")
    sys.stdout.flush()

    try:
        # ||> Step 1.1: Get user selections for extraction
        selected_states = get_user_state_selection()
        selected_years = get_user_year_selection()
        selected_groups = determine_groups_from_years(selected_years)

        # ||> Step 1.2: Get output directory configuration
        title = "📁 Output Directory Configuration:"
        print("\n\n" + "=" * get_visual_width(title))
        print(f"{title}")
        print("=" * get_visual_width(title))
        print(
            "• Files will be organized as: [base_dir]/[state]/extracted/ and [base_dir]/[state]/processed/"
        )
        print("• Directory conflicts will be automatically resolved\n")

        output_dir_prompt = f"Enter output directory (paths can be absolute, or relative to '{parent_dir_of_script_folder}'): "
        user_output_dir = input(output_dir_prompt).strip()

        # ||> ||> Resolve to absolute path
        if not user_output_dir:
            output_dir = parent_dir_of_script_folder
        elif os.path.isabs(user_output_dir):
            output_dir = user_output_dir
        else:
            output_dir = os.path.join(parent_dir_of_script_folder, user_output_dir)
        output_dir = os.path.abspath(output_dir)

        # ||> If user specifies processed/extracted, go up one level
        last_folder = os.path.basename(output_dir).lower()
        if last_folder in ["processed", "extracted"]:
            output_dir = os.path.dirname(output_dir)

        # ||> If now in a state folder, go up one more level
        if is_state_directory(output_dir):
            output_dir = os.path.dirname(output_dir)

        # ||> Ensure output directory exists
        if not ensure_directory_exists(output_dir):
            print("❌ Unable to create output directory. Exiting.")
            sys.exit(1)

        print(f"✅ Base output directory: {output_dir}")

        # ||> Step 1.3: Check for existing files
        file_handling = check_existing_files(
            selected_groups, selected_states, selected_years, output_dir
        )

        if file_handling == "cancel":
            print("Operation cancelled by user.")
            sys.exit(0)

        # ||> ||> Step 1.4: Show extraction summary and confirm
        title = f"🔄 Starting download of {len(selected_groups) * len(selected_states) * len(selected_years)} datasets"
        print("\n\n" + "=" * get_visual_width(title))
        print(f"{title}")
        print("=" * get_visual_width(title))
        selected_cids = get_user_cid_selection()

        # ||> Step 1.5: Get age range and gender analysis configuration
        title = "📊 Age Range and Gender Analysis Configuration:"
        print("\n\n" + "=" * get_visual_width(title))
        print(f"{title}")
        print("=" * get_visual_width(title))

        while True:
            use_age_input = (
                input("Do you want age range analysis? (y/n): ").strip().lower()
            )
            if use_age_input in ["y", "yes"]:
                use_age_ranges = True
                break
            elif use_age_input in ["n", "no"]:
                use_age_ranges = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        age_ranges = None
        if use_age_ranges:
            while True:
                age_input = input(
                    "Enter age ranges (e.g., '15-30,31-45,46-60'): "
                ).strip()
                age_ranges = parse_age_ranges(age_input)
                if age_ranges:
                    print(f"✅ Age ranges: {age_ranges}")
                    break
        else:
            print("✅ Age range analysis: Disabled")

        # ||> Ask about gender analysis
        print()
        while True:
            gender_input = input("Do you want gender analysis? (y/n): ").strip().lower()
            if gender_input in ["y", "yes"]:
                use_gender = True
                break
            elif gender_input in ["n", "no"]:
                use_gender = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        print(f"✅ Gender analysis: {'Enabled' if use_gender else 'Disabled'}")

        # ||> Step 1.6: Ask about output file format
        title = "📄 Output File Format Configuration:"
        print("\n\n" + "=" * get_visual_width(title))
        print(f"{title}")
        print("=" * get_visual_width(title))
        print("Choose how to organize CID codes in output files:")
        print("1. Create separate CSV files for each CID code")
        print("2. Create combined CSV files with all CID codes")
        print("3. Create both separate and combined files")
        print()

        create_separate_files = False
        create_combined_files = False
        
        while True:
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == "1":
                create_separate_files = True
                create_combined_files = False
                print("✅ Will create separate files for each CID code")
                break
            elif choice == "2":
                create_separate_files = False
                create_combined_files = True
                print("✅ Will create combined files with all CID codes")
                break
            elif choice == "3":
                create_separate_files = True
                create_combined_files = True
                print("✅ Will create both separate and combined files")
                break
            else:
                print("Please enter 1, 2, or 3.")

        # ||> Step 1.7: Show complete configuration summary and confirm
        title = "🔄 Complete Pipeline Configuration Summary"
        print("\n\n" + "=" * get_visual_width(title))
        print(f"{title}")
        print("=" * get_visual_width(title))
        print("🔽 EXTRACTION CONFIGURATION:")
        print(f"   • Groups: {selected_groups}")
        print(
            f"   • States: {len(selected_states)} states {selected_states[:3]}{'...' if len(selected_states) > 3 else ''}"
        )
        print(
            f"   • Years: {len(selected_years)} years {selected_years[:5]}{'...' if len(selected_years) > 5 else ''}"
        )
        print(f"   • Output directory: {output_dir}")
        if file_handling != "proceed":
            print(f"   • File handling: {file_handling}")
        print(
            f"   • Total datasets to download: {len(selected_groups) * len(selected_states) * len(selected_years)}"
        )
        print()
        print("🔽 PROCESSING CONFIGURATION:")
        print(f"   • CID codes: {selected_cids}")
        print(f"   • Age range analysis: {'Yes' if use_age_ranges else 'No'}")
        if age_ranges:
            print(f"   • Age ranges: {age_ranges}")
        print(f"   • Gender analysis: {'Yes' if use_gender else 'No'}")
        
        # ||> Show output format configuration
        format_desc = []
        if create_separate_files:
            format_desc.append("separate files per CID")
        if create_combined_files:
            format_desc.append("combined files with all CIDs")
        print(f"   • Output format: {' and '.join(format_desc)}")
        print()

        confirm = (
            input(
                "Proceed with complete pipeline? This may take several minutes (Y/n): "
            )
            .strip()
            .lower()
        )
        print()
        if confirm in ["n", "no"]:
            print("Operation cancelled by user.")
            sys.exit(0)

        downloaded_data, actual_downloads = download_sim_data(
            selected_groups, selected_states, selected_years, output_dir, file_handling
        )

        # ||> Clean up pysus cache
        cleanup_pysus_cache()

        if actual_downloads == 0:
            print("❌ No files were downloaded. Exiting.")
            sys.exit(1)

        # ||> Fetch IBGE municipality data
        ibge_df = fetch_ibge_municipalities()
        if ibge_df is None:
            print("❌ Could not fetch IBGE data. Exiting.")
            sys.exit(1)

        downloaded_files = get_downloaded_files(downloaded_data, output_dir)

        if not downloaded_files:
            print("❌ No downloaded files found for processing.")
            sys.exit(1)

        print(f"📄 Found {len(downloaded_files)} files to process")

        # ||> Validate CID codes against downloaded files
        print(f"\n🔍 Validating CID codes across {len(downloaded_files)} file(s)...")
        print(f"🎯 Looking for CID codes: {selected_cids}")
        valid_files = []

        for file_path in downloaded_files:
            try:
                lf = pl.scan_parquet(file_path)
                if "CAUSABAS" in lf.collect_schema().names():
                    available_cids = (
                        lf.select("CAUSABAS").unique().collect().to_series().to_list()
                    )
                    available_cids = [
                        cid.strip() if isinstance(cid, str) else cid
                        for cid in available_cids
                        if cid is not None
                    ]
                    
                    print(f"\n📋 {os.path.basename(file_path)}: Found {len(available_cids)} unique CIDs in file")
                    
                    valid_cids, invalid_cids = validate_cid_codes(
                        selected_cids, available_cids
                    )

                    if valid_cids:
                        valid_files.append((file_path, valid_cids))
                        print(f"✅ {os.path.basename(file_path)}: Matched {len(valid_cids)} CID(s)")
                    else:
                        print(f"❌ {os.path.basename(file_path)}: No matching CIDs found")
                        if invalid_cids:
                            print(f"   📢  Invalid CIDs: {invalid_cids}")
                else:
                    print(f"❌ {os.path.basename(file_path)}: Missing CAUSABAS column")
            except Exception as e:
                print(f"❌ {os.path.basename(file_path)}: Error reading file - {e}")

        if not valid_files:
            print(
                f"\n❌ None of the downloaded files contain the requested CID codes: {selected_cids}"
            )
            sys.exit(1)

        # ||> Process each file
        successful_files = 0
        failed_files = 0

        for i, (file_path, file_valid_cids) in enumerate(valid_files, 1):
            print(
                f"\n[{i}/{len(valid_files)}] Processing: {os.path.basename(file_path)}"
            )

            try:
                # ||> Get base filename and state
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                if base_name.endswith(".parquet"):
                    base_name = base_name[:-8]

                state_code = extract_state_from_path(file_path)
                file_processed_successfully = False

                # ||> Helper function to save processed file
                def save_processed_file(result_df, output_filename, state_code):
                    if result_df is not None and len(result_df) > 0:
                        if state_code:
                            processed_dir = get_state_aware_directory(
                                output_dir, state_code, "processed"
                            )
                            if not ensure_directory_exists(processed_dir):
                                print(
                                    f"❌ Could not create processed directory for {state_code}"
                                )
                                return False
                            output_path = os.path.join(processed_dir, output_filename)
                        else:
                            output_path = os.path.join(output_dir, output_filename)

                        try:
                            result_df.write_csv(output_path)
                            print(f"✅ Saved: {output_filename}")
                            return True
                        except Exception as e:
                            print(f"❌ Error saving {output_filename}: {e}")
                            return False
                    else:
                        print(f"❌ No data to save for {output_filename}")
                        return False

                # ||> Process separate files for each CID if requested
                if create_separate_files:
                    for cid in file_valid_cids:
                        result_df = process_parquet_file(
                            file_path,
                            age_ranges,
                            use_gender,
                            [cid],
                            ibge_df,
                            aggregate_cids=False,
                        )

                        output_filename = f"{base_name}_P_{cid}.csv"

                        if save_processed_file(result_df, output_filename, state_code):
                            successful_files += 1
                            file_processed_successfully = True
                        else:
                            failed_files += 1
                            print(
                                f"❌ No data for CID {cid} in: {os.path.basename(file_path)}"
                            )

                # ||> Process combined file with all CIDs if requested
                if create_combined_files:
                    result_df = process_parquet_file(
                        file_path,
                        age_ranges,
                        use_gender,
                        file_valid_cids,
                        ibge_df,
                        aggregate_cids=True,
                    )

                    cid_string = "_".join(selected_cids)
                    output_filename = f"{base_name}_P_{cid_string}.csv"

                    if save_processed_file(result_df, output_filename, state_code):
                        successful_files += 1
                        file_processed_successfully = True
                    else:
                        failed_files += 1
                        print(
                            f"❌ Failed to process combined data for: {os.path.basename(file_path)}"
                        )

                if not file_processed_successfully:
                    failed_files += 1

            except Exception as e:
                print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
                failed_files += 1

        total_attempted_extractions = (
            len(selected_groups) * len(selected_states) * len(selected_years)
        )

        title = "🎉 SUS Mortality Data Pipeline Complete!"
        print("\n" + "=" * get_visual_width(title))
        print(f"{title}")
        print("=" * get_visual_width(title))
        print("📥 EXTRACTION PHASE:")
        print(
            f"   ✅ Successful downloads: {actual_downloads}/{total_attempted_extractions}"
        )
        print("📊 PROCESSING PHASE:")
        print(f"   ✅ Successful processing: {successful_files}")
        print(f"   ❌ Failed processing: {failed_files}")
        print(f"📂 Files organized in: {output_dir}")
        print("   • Extracted data: [state]/extracted/")
        print("   • Processed data: [state]/processed/")

        if failed_files > 0:
            print(
                f"\nWARNING: {failed_files} files failed processing. Check error messages above."
            )

        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n🚨  Process interrupted by user.")
        cleanup_pysus_cache()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        cleanup_pysus_cache()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

# %%
