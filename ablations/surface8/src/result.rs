//! 实验结果.

use crate::algos::Profile;
use std::io::{self, Write};

/// 将 `profile` 的结果写进 `w` 中.
fn describe_into<W: Write>(name: &str, p: &Profile, w: &mut W) -> io::Result<()> {
    const S4: &str = "    ";

    #[inline]
    fn f64_to_display(f: Option<f64>) -> String {
        match f {
            Some(f) => format!("{f:.6}"),
            None => "/".to_string(),
        }
    }

    #[inline]
    fn u64_to_display(u: Option<u64>) -> String {
        match u {
            Some(u) => u.to_string(),
            None => "/".to_string(),
        }
    }

    writeln!(w, "Profile `{name}`:")?;
    writeln!(w, "{S4}Invalid backgrounds: {}", p.get_trivial())?;
    writeln!(w, "{S4}Valid foregrounds: {}", p.get_target())?;
    writeln!(w, "{S4}Effective total time: {} us", p.get_target_time_us())?;
    writeln!(
        w,
        "{S4}Effective average time: {} us",
        f64_to_display(p.get_avg_target_time_us())
    )?;
    writeln!(w, "{S4}Total machine time: {} us", p.get_real_time_us())?;
    writeln!(w, "{S4}Eroded in total: {}", p.get_eroded())?;
    writeln!(
        w,
        "{S4}Effective average erosion: {} per image",
        f64_to_display(p.get_avg_eroded())
    )?;
    let t = p.get_most_time_consuming().map(|d| d.as_micros() as u64);
    write!(w, "{S4}Most time-consuming task costs {} us", u64_to_display(t))?;
    Ok(())
}

/// 消融实验最终结果.
pub struct AblationResult {
    data: Vec<(&'static str, Profile)>,
}

impl AblationResult {
    pub fn from_iter<I: IntoIterator<Item = (&'static str, Profile)>>(it: I) -> Self {
        Self {
            data: it.into_iter().collect(),
        }
    }

    /// 分析运行结果.
    pub fn analyze(&self) {
        utils::sep();
        let mut buf = Vec::with_capacity(512);

        for (key, profile) in self.data.iter() {
            describe_into(key, profile, &mut buf).unwrap();
            println!("{}", std::str::from_utf8(&buf).unwrap());
            buf.clear();

            utils::sep();
        }
    }
}
