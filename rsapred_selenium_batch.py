#!/usr/bin/env python3
"""
Batch RSAPred pKd predictions for exported aptamer–molecule pairs.

Microsoft Edge загружает форму RSAPred; отправка — гибридный POST (токен/cookies из
браузера). По умолчанию Edge с окном; headless часто не загружает форму на RSAPred.

Example:
  python rsapred_selenium_batch.py --input rsapred_pairs_to_submit.csv
  python rsapred_selenium_batch.py --headless
  python rsapred_selenium_batch.py --backend requests --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

PREDICT_PAGE = "https://web.iitm.ac.in/bioinfo2/RSAPred/Predict.html"
DEFAULT_RNA_CLASS = "Aptamers"
PKD_PASS_THRESHOLD = 4.5
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
)

_BROWSER_FETCH_JS = """
var callback = arguments[arguments.length - 1];
var rna = arguments[0];
var smiles = arguments[1];
var rnaClass = arguments[2];

try {
    document.dispatchEvent(new MouseEvent('mousemove', {bubbles: true, clientX: 50, clientY: 50}));

    var form = document.getElementById('regForm1');
    if (!form) {
        callback('ERROR:FORM_NOT_FOUND');
        return;
    }

    var sel = document.getElementById('rna_class');
    if (sel) {
        sel.value = rnaClass;
    }
    document.getElementById('rna_seq').value = rna;
    document.getElementById('smiles').value = smiles;

    var fd = new FormData(form);
    fetch(form.action, {method: 'POST', body: fd, credentials: 'same-origin'})
        .then(function(resp) {
            return resp.text().then(function(text) {
                if (!resp.ok) {
                    callback('ERROR:HTTP_' + resp.status + ':' + text.slice(0, 200));
                } else {
                    callback(text);
                }
            });
        })
        .catch(function(err) {
            callback('ERROR:FETCH:' + String(err));
        });
} catch (err) {
    callback('ERROR:JS:' + String(err));
}
"""


def _normalize_rna(seq: str) -> str:
    """Plain RNA sequence (RSAPred accepts without FASTA header)."""
    seq = str(seq).strip().upper().replace("T", "U")
    if seq.startswith(">"):
        lines = [ln.strip() for ln in seq.splitlines() if ln.strip() and not ln.startswith(">")]
        seq = "".join(lines)
    return re.sub(r"\s+", "", seq)


def load_pairs(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        df["pair_id"] = [f"pair_{i}" for i in range(len(df))]

    rna_col = None
    for col in ("rna_sequence", "sequence", "dna_sequence"):
        if col in df.columns:
            rna_col = col
            break
    if rna_col is None:
        raise ValueError("CSV must contain rna_sequence, sequence, or dna_sequence")

    smi_col = None
    for col in ("smiles", "canonical_smiles"):
        if col in df.columns:
            smi_col = col
            break
    if smi_col is None:
        raise ValueError("CSV must contain smiles or canonical_smiles")

    out = pd.DataFrame(
        {
            "pair_id": df["pair_id"].astype(str),
            "rna_sequence": df[rna_col].map(_normalize_rna),
            "smiles": df[smi_col].astype(str).str.strip(),
        }
    )
    for optional in ("sequence_sim", "motif_penalty", "decoded_sim", "rsapred_pkd"):
        if optional in df.columns:
            out[optional] = df[optional]

    if limit is not None:
        out = out.head(int(limit))
    return out.reset_index(drop=True)


def _form_action_url(soup: BeautifulSoup) -> str:
    form = soup.find("form", id="regForm1")
    if form is None:
        raise RuntimeError("RSAPred form (regForm1) not found on predict page")
    action = form.get("action") or "rsapred_php/predict_activity_v2.php"
    return urljoin(PREDICT_PAGE, action)


def parse_result_html(html: str) -> Dict[str, Optional[str]]:
    """Parse RSAPred results table."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="glossary")
    if table is None:
        snippet = re.sub(r"\s+", " ", html[:500]).strip()
        raise ValueError(f"Result table not found. Response head: {snippet[:220]}")

    rows: Dict[str, str] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        key = cells[0].get_text(" ", strip=True)
        val = cells[1].get_text(" ", strip=True)
        rows[key] = val

    pkd_raw = None
    for key, val in rows.items():
        if "Predicted binding affinity" in key:
            pkd_raw = val
            break

    pkd = None
    if pkd_raw:
        m = re.search(r"([-+]?\d*\.?\d+)", pkd_raw.replace(",", "."))
        if m:
            pkd = float(m.group(1))

    units = rows.get("Predicted effective concentration units")
    return {
        "input_rna": rows.get("Input RNA sequence"),
        "input_smiles": rows.get("Input molecule SMILES"),
        "input_rna_category": rows.get("Input RNA category"),
        "rsapred_pkd": pkd,
        "rsapred_units": units,
        "rsapred_pass": (pkd < PKD_PASS_THRESHOLD) if pkd is not None else None,
        "raw_pkd_cell": pkd_raw,
    }


class RequestsRSAPredClient:
    """HTTP POST без браузера."""

    def __init__(self, timeout: int = 120):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": BROWSER_UA,
                "Referer": PREDICT_PAGE,
            }
        )
        self.timeout = timeout

    def predict(self, rna_sequence: str, smiles: str, rna_class: str = DEFAULT_RNA_CLASS) -> Dict:
        page = self.session.get(PREDICT_PAGE, timeout=self.timeout)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        nc = soup.find("input", {"name": "__ncforminfo"})
        if nc is None or not nc.get("value"):
            raise RuntimeError("Could not read RSAPred anti-bot token (__ncforminfo)")

        post_url = _form_action_url(soup)
        data = {
            "rna_class": rna_class,
            "rna_seq": rna_sequence,
            "smiles": smiles,
            "ncformfield": "",
            "__ncforminfo": nc["value"],
        }
        resp = self.session.post(post_url, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return parse_result_html(resp.text)


def _format_exc(exc: Exception) -> str:
    msg = str(exc).strip()
    if msg:
        return msg.split("\n", maxsplit=1)[0][:300]
    return f"{type(exc).__name__} (timeout or empty Selenium message)"


class SeleniumRSAPredClient:
    """Edge загружает форму; POST через requests с cookies/токеном из браузера."""

    def __init__(
        self,
        headless: bool = False,
        driver_path: Optional[str] = None,
        timeout: int = 120,
    ):
        self.headless = headless
        self.driver_path = driver_path
        self.timeout = timeout
        self.driver = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _restart_browser(self, headless: bool) -> None:
        self.close()
        self.headless = headless
        label = "headless" if headless else "с окном"
        print(f"Запуск Edge ({label}, Selenium Manager)...", flush=True)
        self.driver = create_edge_driver(
            driver_path=self.driver_path,
            headless=headless,
            timeout=self.timeout,
        )

    def _start(self) -> None:
        self._restart_browser(self.headless)
        try:
            self._open_predict_page()
            print("Страница RSAPred загружена (прогрев Edge).", flush=True)
        except Exception as exc:
            if self.headless:
                print("Headless не загрузил форму — переключаюсь на Edge с окном...", flush=True)
                try:
                    self._restart_browser(headless=False)
                    self._open_predict_page()
                    print("Страница RSAPred загружена (Edge с окном).", flush=True)
                    return
                except Exception as exc2:
                    print(f"Прогрев не удался ({_format_exc(exc2)}) — повторим на паре.", flush=True)
                    return
            print(f"Прогрев страницы не удался ({_format_exc(exc)}) — повторим на паре.", flush=True)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    def _page_has_form(self) -> bool:
        assert self.driver is not None
        try:
            src = self.driver.page_source or ""
        except Exception:
            return False
        return "regForm1" in src and "__ncforminfo" in src

    def _open_predict_page(self) -> None:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        assert self.driver is not None
        last_exc: Optional[Exception] = None

        for attempt in range(1, 3):
            try:
                self.driver.get(PREDICT_PAGE)
            except TimeoutException as exc:
                last_exc = exc
                print("    Таймаут page load — жду форму в DOM...", flush=True)

            deadline = time.time() + 60
            while time.time() < deadline:
                if self._page_has_form():
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.NAME, "__ncforminfo"))
                        )
                    except Exception:
                        pass
                    self.driver.execute_script(
                        "document.dispatchEvent(new MouseEvent('mousemove', "
                        "{bubbles:true, clientX:50, clientY:50}));"
                    )
                    return
                time.sleep(1)

            last_exc = RuntimeError("Форма regForm1 не появилась за 60 с")
            if attempt < 2:
                print(f"    Повтор открытия страницы ({attempt}/2)...", flush=True)
                time.sleep(2)

        raise last_exc or RuntimeError("Не удалось загрузить форму RSAPred (regForm1)")

    def _session_from_browser(self) -> tuple[requests.Session, str, str]:
        from selenium.webdriver.common.by import By

        assert self.driver is not None
        nc_el = self.driver.find_element(By.NAME, "__ncforminfo")
        nc_value = nc_el.get_attribute("value")
        if not nc_value:
            raise RuntimeError("Could not read RSAPred token (__ncforminfo)")

        post_url = self.driver.execute_script(
            "return document.getElementById('regForm1').action;"
        )
        if not post_url:
            raise RuntimeError("Could not read RSAPred form action URL")

        user_agent = self.driver.execute_script("return navigator.userAgent;")
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": user_agent or BROWSER_UA,
                "Referer": PREDICT_PAGE,
            }
        )
        jar = requests.cookies.RequestsCookieJar()
        for cookie in self.driver.get_cookies():
            jar.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )
        session.cookies = jar
        return session, post_url, nc_value

    def _submit_via_hybrid_post(
        self, rna_sequence: str, smiles: str, rna_class: str
    ) -> Dict:
        session, post_url, nc_value = self._session_from_browser()
        data = {
            "rna_class": rna_class,
            "rna_seq": rna_sequence,
            "smiles": smiles,
            "ncformfield": "",
            "__ncforminfo": nc_value,
        }
        resp = session.post(post_url, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return parse_result_html(resp.text)

    def _submit_via_browser_fetch(self, rna_sequence: str, smiles: str, rna_class: str) -> str:
        assert self.driver is not None
        html = self.driver.execute_async_script(
            _BROWSER_FETCH_JS,
            rna_sequence,
            smiles,
            rna_class,
        )
        if not isinstance(html, str):
            raise RuntimeError(f"Unexpected browser response type: {type(html)}")
        if html.startswith("ERROR:"):
            raise RuntimeError(html)
        return html

    def _predict_once(self, rna_sequence: str, smiles: str, rna_class: str) -> Dict:
        self._open_predict_page()
        try:
            print("    POST (токен/cookies из Edge)...", flush=True)
            return self._submit_via_hybrid_post(rna_sequence, smiles, rna_class)
        except Exception as hybrid_exc:
            print(f"    Hybrid POST не удался ({_format_exc(hybrid_exc)}), пробую fetch...", flush=True)
            html = self._submit_via_browser_fetch(rna_sequence, smiles, rna_class)
            return parse_result_html(html)

    def predict(self, rna_sequence: str, smiles: str, rna_class: str = DEFAULT_RNA_CLASS) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                return self._predict_once(rna_sequence, smiles, rna_class)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    print(f"    Повтор ({attempt}/2): {_format_exc(exc)}", flush=True)
                    time.sleep(2)
        raise RuntimeError(_format_exc(last_exc or RuntimeError("RSAPred failed")))


def create_edge_driver(
    driver_path: Optional[str] = None,
    headless: bool = False,
    timeout: int = 120,
):
    """Создаёт Edge WebDriver (Selenium Manager подбирает msedgedriver)."""
    from selenium import webdriver
    from selenium.common.exceptions import SessionNotCreatedException
    from selenium.webdriver.edge.service import Service as EdgeService

    options = webdriver.EdgeOptions()
    options.page_load_strategy = "eager"
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    if driver_path:
        try:
            print(f"Запуск Edge с драйвером: {driver_path}")
            driver = webdriver.Edge(service=EdgeService(driver_path), options=options)
        except SessionNotCreatedException as exc:
            if "only supports Microsoft Edge version" in str(exc):
                print(
                    "Версия msedgedriver не совпадает с Edge. "
                    "Пробую автоматический подбор драйвера (Selenium Manager)..."
                )
                driver = webdriver.Edge(options=options)
            else:
                raise
    else:
        driver = webdriver.Edge(options=options)

    driver.set_page_load_timeout(timeout)
    driver.set_script_timeout(timeout)
    return driver


def run_batch(
    pairs: pd.DataFrame,
    output_csv: str,
    backend: str = "selenium",
    headless: bool = False,
    driver_path: Optional[str] = None,
    delay_sec: float = 2.0,
    resume: bool = True,
    dry_run: bool = False,
    timeout: int = 120,
) -> pd.DataFrame:
    output_path = Path(output_csv)
    done_ids: set = set()
    results: List[Dict] = []

    if resume and output_path.exists():
        prev = pd.read_csv(output_path)
        ok = prev[prev["status"] == "ok"]
        done_ids = set(ok["pair_id"].astype(str))
        results = ok.to_dict("records")
        print(f"Resume: skipping {len(done_ids)} completed pairs")

    pending = pairs[~pairs["pair_id"].astype(str).isin(done_ids)]
    print(f"Pairs to run: {len(pending)} / {len(pairs)}")

    if dry_run:
        for _, row in pending.iterrows():
            print(f"  [dry-run] {row['pair_id']}: {row['smiles'][:40]}... | RNA len={len(row['rna_sequence'])}")
        return pd.DataFrame(results)

    if backend == "requests":
        client = RequestsRSAPredClient(timeout=timeout)
        predict_fn = client.predict
        context = None
    elif backend == "selenium":
        context = SeleniumRSAPredClient(
            headless=headless,
            driver_path=driver_path,
            timeout=timeout,
        )
        context._start()
        predict_fn = context.predict
    else:
        raise ValueError(f"Unknown backend: {backend}")

    try:
        for n, (_, row) in enumerate(pending.iterrows(), start=1):
            pair_id = str(row["pair_id"])
            print(f"[{n}/{len(pending)}] {pair_id} ...", flush=True)
            record = {
                "pair_id": pair_id,
                "rna_sequence": row["rna_sequence"],
                "smiles": row["smiles"],
                "backend": backend,
                "predicted_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            }
            for col in ("sequence_sim", "motif_penalty", "decoded_sim"):
                if col in row and pd.notna(row[col]):
                    record[col] = row[col]

            try:
                parsed = predict_fn(row["rna_sequence"], row["smiles"])
                record.update(parsed)
                record["status"] = "ok"
                pkd = parsed.get("rsapred_pkd")
                print(f"    pKd={pkd} pass={parsed.get('rsapred_pass')} units={parsed.get('rsapred_units')}")
            except Exception as exc:  # noqa: BLE001 — batch must continue
                record["status"] = "error"
                record["error"] = _format_exc(exc)
                print(f"    ERROR: {record['error']}")

            results.append(record)
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")

            if delay_sec > 0 and n < len(pending):
                time.sleep(delay_sec)
    finally:
        if backend == "selenium" and context is not None:
            context.close()

    out_df = pd.DataFrame(results)
    if len(out_df):
        ok_n = (out_df["status"] == "ok").sum()
        if "rsapred_pass" in out_df.columns:
            pass_n = int(out_df.loc[out_df["status"] == "ok", "rsapred_pass"].eq(True).sum())
        else:
            pass_n = 0
        print(f"\nSaved: {output_path} ({len(out_df)} rows, ok={ok_n}, pass={pass_n})")
    return out_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch RSAPred predictions via Selenium or HTTP.")
    p.add_argument(
        "--input",
        default="rsapred_pairs_to_submit.csv",
        help="Input CSV with pair_id, RNA, SMILES",
    )
    p.add_argument(
        "--output",
        default="rsapred_automation_results.csv",
        help="Output CSV (appended/resumed)",
    )
    p.add_argument(
        "--backend",
        choices=("selenium", "requests"),
        default="selenium",
        help="selenium=Edge browser (default); requests=HTTP POST without browser",
    )
    p.add_argument(
        "--driver",
        default=None,
        help="Путь к msedgedriver.exe (необязательно; иначе Selenium Manager)",
    )
    p.add_argument("--headless", action="store_true", help="Headless Edge (на RSAPred часто не работает)")
    p.add_argument("--delay", type=float, default=2.5, help="Seconds between pairs")
    p.add_argument("--limit", type=int, default=None, help="Max pairs to process")
    p.add_argument("--timeout", type=int, default=120, help="Page/script timeout (sec)")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing output file")
    p.add_argument("--dry-run", action="store_true", help="Print pairs only")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    pairs = load_pairs(args.input, limit=args.limit)
    print(f"Loaded {len(pairs)} pairs from {args.input}")

    run_batch(
        pairs=pairs,
        output_csv=args.output,
        backend=args.backend,
        headless=args.headless,
        driver_path=args.driver,
        delay_sec=args.delay,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
