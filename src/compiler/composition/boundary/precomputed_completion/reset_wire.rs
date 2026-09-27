// Bounded, source-certified wire form for the component-only reset observer.
// Query metadata, owner IDs and composition peers are deliberately absent.
const MAGIC:&[u8;8]=b"GLRRFX01";
const MAX_WIRE:usize=64*1024*1024;
impl ResetIndex{
    pub fn to_bytes(&self)->Option<Vec<u8>>{
        fn put(out:&mut Vec<u8>,n:usize)->Option<()>{out.extend_from_slice(&u32::try_from(n).ok()?.to_le_bytes());Some(())}
        fn row(out:&mut Vec<u8>,ids:&[u32])->Option<()>{put(out,ids.len())?;for &id in ids{put(out,id as usize)?;}Some(())}
        let cells=self.nodes.iter().try_fold(0usize,|n,r|n.checked_add(r.frontier.len())?.checked_add(r.matched.len())?.checked_add(r.future.len())?.checked_add(r.edges.len().checked_mul(2)?)?.checked_add(4))?;
        let bytes=cells.checked_mul(4)?.checked_add(20)?;if bytes>MAX_WIRE{return None}
        let mut out=Vec::with_capacity(bytes);out.extend_from_slice(MAGIC);put(&mut out,self.source.num_states()as usize)?;put(&mut out,self.source.num_terminals()as usize)?;put(&mut out,self.nodes.len())?;
        for node in &self.nodes{
            row(&mut out,&node.frontier)?;row(&mut out,&node.matched)?;row(&mut out,&node.future)?;
            put(&mut out,node.edges.len())?;for(&b,&q)in &node.edges{put(&mut out,b as usize)?;put(&mut out,q as usize)?;}
        }
        debug_assert_eq!(out.len(),bytes);Some(out)
    }
    pub fn from_bytes(source:Arc<Tokenizer>,bytes:&[u8])->Option<Self>{
        if bytes.len()>MAX_WIRE||!bytes.starts_with(MAGIC)||source.num_states()==0||source.num_states()>500_000||source.has_virtual_residual_runtime(){return None}
        struct Reader<'a>{bytes:&'a[u8],at:usize}
        impl Reader<'_>{
            fn word(&mut self)->Option<usize>{let end=self.at.checked_add(4)?;let value=u32::from_le_bytes(self.bytes.get(self.at..end)?.try_into().ok()?);self.at=end;Some(value as usize)}
            fn ids(&mut self,budget:&mut usize,max:u32)->Option<Vec<u32>>{
                let count=self.word()?;if count>*budget{return None}*budget-=count;
                let end=self.at.checked_add(count.checked_mul(4)?)?;let raw=self.bytes.get(self.at..end)?;self.at=end;
                let mut result=Vec::with_capacity(count);let mut prev=None;
                for bytes in raw.chunks_exact(4){let id=u32::from_le_bytes(bytes.try_into().ok()?);if id>=max||prev.is_some_and(|p|p>=id){return None}prev=Some(id);result.push(id);}
                Some(result)
            }
        }
        let mut r=Reader{bytes,at:8};let raw=r.word()?;let terms=r.word()?;let count=r.word()?;
        if raw!=source.num_states()as usize||terms!=source.num_terminals()as usize||count==0||count>100_000||count.checked_mul(16)?>bytes.len().saturating_sub(r.at){return None}
        let(mut frontier_left,mut labels_left,mut edge_left)=(2_000_000usize,8_000_000usize,1_000_000usize);
        let mut nodes=Vec::with_capacity(count);let mut stats=Stats::default();
        for _ in 0..count{
            let frontier=r.ids(&mut frontier_left,raw as u32)?;if frontier.is_empty(){return None}
            let matched=r.ids(&mut labels_left,terms as u32)?;let future=r.ids(&mut labels_left,terms as u32)?;
            let edges_count=r.word()?;if edges_count>256||edges_count>edge_left{return None}edge_left-=edges_count;
            if edges_count.checked_mul(8)?>bytes.len().saturating_sub(r.at){return None}
            let mut edges=BTreeMap::new();let mut prev=None;
            for _ in 0..edges_count{let b=r.word()?;let target=r.word()?;
                if b>255||prev.is_some_and(|p|p>=b)||(target!=DEAD as usize&&target>=count){return None}prev=Some(b);edges.insert(b as u8,target as u32);
            }
            stats.frontier_cells+=frontier.len();stats.labels+=matched.len()+future.len();stats.edges+=edges_count;
            nodes.push(Node{frontier,matched,future,edges});
        }
        if r.at!=bytes.len(){return None}stats.nodes=count;let index=Self{source,nodes,stats};
        index.verify().then_some(index)
    }
}
