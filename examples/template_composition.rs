//! Public precompiled template composition; see docs/template-parser-composition.md.
fn main() -> glrmask::Result<()> {
    use glrmask::{BuildOptions, Constraint, Grammar, Optimization, ParserBackend, Vocab};
    
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"y".to_vec()),
        (3, b"xay".to_vec()), (4, b"xy".to_vec()),
    ]);
    let child = Grammar::from_ebnf(r#"start ::= "a"?"#).compile_with(
        &vocab,
        BuildOptions::default()
            .optimization(Optimization::FastRuntime)
            .parser_backend(ParserBackend::TemplateDfa),
    )?;
    let parent = Grammar::from_glrm(
        r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#,
    ).compile_unlinked(&vocab)?;
    
    for optimization in [Optimization::FastBuild, Optimization::FastRuntime] {
        let linked = parent.bind("child", &child)?.link_with(
            BuildOptions::default()
                .optimization(optimization)
                .parser_backend(ParserBackend::TemplateDfa),
        )?;
        assert_eq!(linked.parser_backend(), ParserBackend::TemplateDfa);
        let mut state = linked.start();
        state.commit_token(4)?; // The child body can be empty.
        assert!(state.is_accepting());
    
        let loaded = Constraint::load(linked.save())?;
        let mut state = loaded.start();
        state.commit_token(3)?;
        assert!(state.is_accepting());
    }
    Ok(())
}
