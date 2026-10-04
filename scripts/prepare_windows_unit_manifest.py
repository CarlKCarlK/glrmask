"""Prepare a portable, isolated unit-only Cargo graph; never edit canonical files.

Requires Python 3.11+. --check-only validates/prints the proposed manifest without
creating files or invoking Cargo. Actual preparation resolves offline, checks all
package versions against the canonical lock, then requires --locked metadata.
"""
import argparse,hashlib,json,os,re,shutil,subprocess,tomllib
from pathlib import Path

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def section(text,name):
    match=re.search(r'(?ms)^\['+re.escape(name)+r'\]\s*\n(.*?)(?=^\[|\Z)',text)
    if not match:raise ValueError('Required ordinary section missing: '+name)
    return match.group(1)
def versions(lock):
    return {(p['name'],p['version'],p.get('source')) for p in tomllib.loads(lock)['package']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();repo=args.repo.resolve();directory=(args.directory or repo/'.cache/windows-unit').resolve()
    assert directory.is_relative_to(repo), 'Use an owned preparation directory inside this checkout'
    assert directory!=repo and not directory.is_relative_to(repo/'src')
    manifest=repo/'Cargo.toml';lock=repo/'Cargo.lock';before={str(p):sha(p) for p in (manifest,lock)}
    text=manifest.read_text(encoding='utf-8');canonical=tomllib.loads(text);package=canonical['package']
    assert not package.get('build') and not (repo/'build.rs').exists(), 'Root build script needs separate path review'
    assert 'internal-api' in canonical['features']
    assert (repo/'src/lib.rs').exists()
    # This checkout uses ordinary inline dependency tables. Fail closed rather
    # than silently dropping a dependency subsection or workspace inheritance.
    assert not re.search(r'^\[(?:dev-)?dependencies\.',text,re.M)
    deps=section(text,'dependencies');dev=section(text,'dev-dependencies')
    omitted=[]
    for alias,value in canonical.get('dev-dependencies',{}).items():
        if isinstance(value,dict) and value.get('package',alias)==package['name'] and 'path' in value and (repo/value['path']).resolve()==repo:
            pattern=r'^'+re.escape(alias)+r'\s*=.*(?:\n|\Z)'
            dev,count=re.subn(pattern,'',dev,flags=re.M);assert count==1
            omitted.append(alias)
    assert omitted, 'Review self dev-dependency inventory for this checkout'
    def rebase(match):
        local=(repo/match.group(1)).resolve();assert local.exists()
        return 'path = '+json.dumps(local.as_posix())
    deps=re.sub(r'\bpath\s*=\s*"([^"\n]+)"',rebase,deps)
    dev=re.sub(r'\bpath\s*=\s*"([^"\n]+)"',rebase,dev)
    unit='[package]\n'+''.join(k+' = '+json.dumps(str(package[k]))+'\n' for k in ('name','version','edition'))
    unit+='autotests = false\nautoexamples = false\nautobenches = false\nautobins = false\n'
    unit+='\n[features]\n'+section(text,'features')
    unit+='\n[lib]\npath = '+json.dumps((repo/'src/lib.rs').as_posix())+'\n'
    unit+='\n[dependencies]\n'+deps+'\n[dev-dependencies]\n'+dev+'\n[workspace]\nresolver = "2"\n'
    for profile in ('release','bench'):
        unit+='\n[profile.'+profile+']\nopt-level = 3\ncodegen-units = 16\nincremental = true\ndebug = 0\nlto = false\ndebug-assertions = false\noverflow-checks = false\n'
    parsed=tomllib.loads(unit);assert omitted[0] not in parsed.get('dev-dependencies',{})
    identity=dict(repo=str(repo),canonical_hashes=before,omitted_isolated_self_dev_dependencies=omitted,
        unit_manifest_sha256=hashlib.sha256(unit.encode()).hexdigest(),manifest=str(directory/'Cargo.toml'),
        target=str(repo/'target/windows-unit'),features=['default','internal-api'],check_only=args.check_only,
        canonical_files_unchanged=True,root_only_lto_off_extra_argument_required=True)
    if not args.check_only:
        directory.mkdir(parents=True,exist_ok=True);unit_path=directory/'Cargo.toml';unit_lock=directory/'Cargo.lock'
        if unit_path.exists():assert unit_path.read_text(encoding='utf-8')==unit,'Existing bridge differs; inspect rather than overwrite'
        else:
            with unit_path.open('x',encoding='utf-8',newline='\n') as handle:handle.write(unit)
        if not unit_lock.exists():shutil.copy2(lock,unit_lock)
        env=os.environ.copy();env.update(RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='',CARGO_BUILD_JOBS='2')
        for key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS'):env.pop(key,None)
        cargo=shutil.which('cargo');assert cargo,'Cargo must be on PATH'
        base=[cargo,'metadata','--offline','--manifest-path',str(unit_path),'--features','internal-api','--format-version','1']
        subprocess.run(base,env=env,cwd=repo,check=True,capture_output=True,timeout=60)
        added=versions(unit_lock.read_text(encoding='utf-8'))-versions(lock.read_text(encoding='utf-8'))
        assert not added, 'Offline resolution introduced package versions; inspect generated lock before compiling: '+repr(added)
        subprocess.run(base+['--locked'],env=env,cwd=repo,check=True,capture_output=True,timeout=60)
        identity['unit_lock_sha256']=sha(unit_lock)
    assert all(sha(Path(path))==digest for path,digest in before.items())
    print(json.dumps(identity,indent=2))

if __name__=='__main__':main()
